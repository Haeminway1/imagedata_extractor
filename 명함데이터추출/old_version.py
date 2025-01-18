import os
import base64
import json
import yaml
import anthropic
import pandas as pd
import cv2
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime

class ClaudeImageExtractor:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Claude Image Extractor with configuration file.
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.client = anthropic.Client(api_key=self.config['api_key'])
        self._setup_directories()
        self.MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB in bytes

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _setup_directories(self):
        """Create output directory if it doesn't exist."""
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)

    def optimize_image_size(self, image: np.ndarray, max_size: int = None) -> np.ndarray:
        """
        Optimize image size to meet API requirements.
        
        Args:
            image (np.ndarray): Input image
            max_size (int): Maximum size in bytes (default: 5MB)
            
        Returns:
            np.ndarray: Optimized image
        """
        if max_size is None:
            max_size = self.MAX_IMAGE_SIZE

        # Initial quality and size parameters
        quality = 95
        max_dimension = 2000
        min_quality = 30
        min_dimension = 800

        # First try: reduce dimensions if too large
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Resized image to {new_width}x{new_height}")

        while True:
            # Encode image
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            size = len(buffer)
            
            # Check if size is acceptable
            if size <= max_size:
                print(f"Final image size: {size/1024/1024:.2f}MB, Quality: {quality}")
                return image

            # Reduce quality first
            if quality > min_quality:
                quality -= 5
                print(f"Reducing quality to {quality}")
                continue

            # If quality is at minimum, reduce dimensions
            height, width = image.shape[:2]
            if min(height, width) <= min_dimension:
                print(f"Warning: Cannot reduce image further while maintaining minimum quality")
                return image

            scale = 0.8  # Reduce dimensions by 20%
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            quality = 95  # Reset quality for the new size
            print(f"Reduced dimensions to {new_width}x{new_height}")

    def encode_image(self, image: np.ndarray) -> str:
        """
        Encode image to base64 string with size optimization.
        """
        try:
            # Optimize image size
            optimized_image = self.optimize_image_size(image)
            
            # Encode with reduced quality if needed
            for quality in [95, 85, 75, 65]:
                _, buffer = cv2.imencode('.jpg', optimized_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                size = len(buffer)
                
                if size <= self.MAX_IMAGE_SIZE:
                    print(f"Encoded image size: {size/1024/1024:.2f}MB (quality: {quality})")
                    return base64.b64encode(buffer).decode('utf-8')
            
            # If still too large, return error
            raise ValueError("Could not reduce image to acceptable size")
            
        except Exception as e:
            raise ValueError(f"Error encoding image: {str(e)}")

    def detect_business_card_simple(self, image: np.ndarray) -> np.ndarray:
        """
        Improved business card detection that handles both cropped and uncropped images.
        """
        try:
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Check if image is already cropped (close to business card dimensions)
            aspect_ratio = width / height
            if 1.4 <= aspect_ratio <= 2.2 and width * height < 1000000:
                print("Image appears to be pre-cropped, skipping detection")
                return image
                
            # Apply adaptive thresholding
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Sort contours by area, largest first
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            best_card = None
            max_score = 0
            
            for contour in contours[:5]:  # Check only largest 5 contours
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip if too small
                if w * h < 0.01 * width * height:
                    continue
                    
                aspect_ratio = w / h
                area_ratio = (w * h) / (width * height)
                
                # Scoring system
                score = 0
                
                # Aspect ratio score
                if 1.5 <= aspect_ratio <= 1.8:  # Ideal business card ratio
                    score += 3
                elif 1.2 <= aspect_ratio <= 2.2:  # Acceptable ratio
                    score += 1
                    
                # Size score
                if 0.1 <= area_ratio <= 0.5:
                    score += 2
                    
                # Position score (prefer lower half for notebook images)
                relative_y = y / height
                if 0.5 <= relative_y <= 0.9:
                    score += 2
                    
                # Right side preference for notebook images
                relative_x = (x + w/2) / width
                if relative_x >= 0.5:
                    score += 1
                    
                if score > max_score:
                    max_score = score
                    best_card = (x, y, w, h)
            
            if best_card is None:
                print("No card detected, returning original image")
                return image
                
            # Extract card with padding
            x, y, w, h = best_card
            padding = int(min(w, h) * 0.05)  # 5% padding
            
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + w + padding)
            y2 = min(height, y + h + padding)
            
            card_image = image[y1:y2, x1:x2]
            
            # Enhance contrast
            lab = cv2.cvtColor(card_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            card_image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Save debug image
            debug_dir = Path("debug_images")
            debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_dir / "detected_card.jpg"), card_image)
            
            return card_image
            
        except Exception as e:
            print(f"Error in card detection: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return image

    def _call_claude_api(self, base64_image: str) -> Dict:
        """Call Claude API with the image and parse the response."""
        try:
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0,
                system="Identify the business card in the picture and Extract the following information from business card images:\n"
                       "The business card is on the bottom right corner on a notebook.\n"
                       "- Name (in both original language and English if available)\n"
                       "- Title/Position\n"
                       "- Company Name (in both original language and English if available)\n"
                       "- Address (both office and factory if available)\n"
                       "- Contact Information:\n"
                       "  * Mobile phone\n"
                       "  * Office phone\n"
                       "  * Fax\n"
                       "  * Email\n"
                       "  * Website\n"
                       "- EXPO information if available\n"
                       "- Any additional notes\n\n"
                       "Format all information exactly as shown on the card.\n"
                       "Do not modify or reformat phone numbers, addresses, or other data.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Extract and return the business card information in this exact JSON format:
                                    {
                                        "name": {
                                            "original": "",
                                            "english": ""
                                        },
                                        "title": "",
                                        "company": {
                                            "original": "",
                                            "english": ""
                                        },
                                        "address": {
                                            "office": "",
                                            "factory": ""
                                        },
                                        "contact": {
                                            "mobile": "",
                                            "office_phone": "",
                                            "fax": "",
                                            "email": "",
                                            "website": ""
                                        },
                                        "expo": "",
                                        "additional_notes": ""
                                    }"""
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            )
            
            response_text = message.content[0].text.strip()
            return self._parse_json_response(response_text)
                
        except Exception as e:
            print(f"Error calling Claude API: {str(e)}")
            return {"error": f"API call failed: {str(e)}"}

    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON from Claude's response."""
        try:
            # Clean up the response text
            response_text = response_text.strip()
            
            # Remove any markdown code block indicators
            response_text = response_text.replace('```json', '').replace('```', '')
            
            # Try direct JSON parsing
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"Initial JSON parsing failed: {str(e)}")
            
            # Try to extract JSON portion using regex
            import re
            json_pattern = r'\{(?:[^{}]|(?R))*\}'
            matches = re.finditer(json_pattern, response_text, re.DOTALL)
            
            largest_match = None
            max_length = 0
            
            for match in matches:
                if len(match.group()) > max_length:
                    largest_match = match.group()
                    max_length = len(match.group())
            
            if largest_match:
                try:
                    return json.loads(largest_match)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse extracted JSON portion: {str(e)}")
            
            print("\nFailed to parse response. Original response:")
            print(response_text)
            return {"error": "Could not parse JSON from response"}
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return {"error": f"Error parsing response: {str(e)}"}

    def process_single_image(self, image_path: str) -> Dict:
        """
        Process a single image with improved error handling.
        """
        try:
            print(f"\nProcessing image: {image_path}")
            
            # Read image
            with open(image_path, 'rb') as file:
                img_array = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
            if image is None:
                raise ValueError("Failed to read image")
            
            # Print debug info
            print(f"Original image shape: {image.shape}")
            original_size = os.path.getsize(image_path)
            print(f"Original file size: {original_size/1024/1024:.2f}MB")
            
            # Optimize large images first
            if original_size > self.MAX_IMAGE_SIZE:
                print("Optimizing large image...")
                image = self.optimize_image_size(image)
            
            # Process image
            card_image = self.detect_business_card_simple(image)
            print(f"Processed image shape: {card_image.shape}")
            
            # Save debug image
            debug_dir = Path("debug_images")
            debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_dir / f"processed_{Path(image_path).name}"), card_image)
            
            # Convert to base64
            print("Converting to base64...")
            base64_image = self.encode_image(card_image)
            
            # Call Claude API
            print("Calling Claude API...")
            result = self._call_claude_api(base64_image)
            print("API call completed")
            
            return result
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {"error": str(e)}

    def process_images(self) -> List[Dict]:
        """Process all images in the input directory."""
        results = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        
        # Get list of all image files
        image_files = [
            f for f in Path(self.config['input_dir']).iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"Found {len(image_files)} image files")
        
        for i, image_file in enumerate(image_files, 1):
            result = self.process_single_image(str(image_file))
            results.append({
                "filename": image_file.name,
                "data": result,
                "processed_at": datetime.now().isoformat()
            })
        
        # Save results
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save to JSON
        json_path = output_dir / self.config['json_output']
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        # Convert to Excel
        self._save_to_excel(results)
        
        print(f"\nProcessing complete!")
        print(f"Results saved to:")
        print(f"- JSON: {json_path}")
        print(f"- Excel: {output_dir / self.config['excel_output']}")
        
        return results

    def _save_to_excel(self, results: List[Dict]):
        """Save results to Excel file with proper formatting."""
        # Flatten the nested dictionaries
        flattened_data = []
        for result in results:
            flat_item = {
                'filename': result['filename'],
                'processed_at': result['processed_at']
            }
            
            # Extract data
            data = result['data']
            if 'error' in data:
                flat_item['error'] = data['error']
                flattened_data.append(flat_item)
                continue
            
            # Flatten nested structure with more intuitive column names
            flat_data = {
                'name_original': data.get('name', {}).get('original', ''),
                'name_english': data.get('name', {}).get('english', ''),
                'title': data.get('title', ''),
                'company_original': data.get('company', {}).get('original', ''),
                'company_english': data.get('company', {}).get('english', ''),
                'address_office': data.get('address', {}).get('office', ''),
                'address_factory': data.get('address', {}).get('factory', ''),
                'mobile_phone': data.get('contact', {}).get('mobile', ''),
                'office_phone': data.get('contact', {}).get('office_phone', ''),
                'fax': data.get('contact', {}).get('fax', ''),
                'email': data.get('contact', {}).get('email', ''),
                'website': data.get('contact', {}).get('website', ''),
                'expo': data.get('expo', ''),
                'additional_notes': data.get('additional_notes', '')
            }
            
            flat_item.update(flat_data)
            flattened_data.append(flat_item)
        
        # Convert to DataFrame
        df = pd.DataFrame(flattened_data)
        
        # Define column order
        columns = [
            'filename', 
            'processed_at',
            'name_original',
            'name_english',
            'title',
            'company_original',
            'company_english',
            'address_office',
            'address_factory',
            'mobile_phone',
            'office_phone',
            'fax',
            'email',
            'website',
            'expo',
            'additional_notes'
        ]
        
        # Add error column if exists
        if 'error' in df.columns:
            columns.append('error')
        
        # Reorder columns and handle any extra columns that might exist
        existing_columns = [col for col in columns if col in df.columns]
        extra_columns = [col for col in df.columns if col not in columns]
        df = df[existing_columns + extra_columns]
        
        # Save to Excel with formatting
        excel_path = Path(self.config['output_dir']) / self.config['excel_output']
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Extracted Data')
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Extracted Data']
            for i, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                ) + 2
                worksheet.column_dimensions[worksheet.cell(1, i+1).column_letter].width = max_length
                
            # Add column headers in Korean
            korean_headers = {
                'filename': '파일명',
                'processed_at': '처리일시',
                'name_original': '이름(원어)',
                'name_english': '이름(영문)',
                'title': '직위',
                'company_original': '회사명(원어)',
                'company_english': '회사명(영문)',
                'address_office': '사무실주소',
                'address_factory': '공장주소',
                'mobile_phone': '휴대전화',
                'office_phone': '사무실전화',
                'fax': '팩스',
                'email': '이메일',
                'website': '웹사이트',
                'expo': '전시회정보',
                'additional_notes': '추가메모',
                'error': '오류'
            }
            
            for i, column in enumerate(df.columns, start=1):
                cell = worksheet.cell(1, i)
                cell.value = korean_headers.get(column, column)

def main():
    """Main function to run the business card extraction process."""
    try:
        # Initialize extractor with config file
        extractor = ClaudeImageExtractor("config.yaml")
        
        # Process all images
        results = extractor.process_images()
        
        print(f"\nProcessing complete!")
        print(f"Results saved to:")
        print(f"- JSON: {Path(extractor.config['output_dir']) / extractor.config['json_output']}")
        print(f"- Excel: {Path(extractor.config['output_dir']) / extractor.config['excel_output']}")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()