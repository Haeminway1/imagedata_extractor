import os
import re
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
    def _clean_phone_number(self, phone: str) -> str:
    
        if not phone:
            return ""
        
        # Remove common prefixes while preserving country codes
        phone = re.sub(r'^(Tel|Phone|Mobile|Fax)\s*:?\s*', '', phone, flags=re.IGNORECASE)
        
        # Standardize format but keep original structure
        phone = re.sub(r'\s+', ' ', phone).strip()
        
        # Preserve country code format
        if phone.startswith('+'):
            return phone
        elif re.match(r'^\d{1,4}-', phone):
            return f"+{phone}"
        elif phone.startswith('0'):
            return phone
        else:
            return phone

    def _clean_email(self, email: str) -> str:
        """Clean and validate email addresses."""
        if not email:
            return ""
        
        # Remove common prefixes
        email = re.sub(r'^E?-?mail\s*:?\s*', '', email, flags=re.IGNORECASE)
        
        # Extract email from common patterns
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        matches = re.findall(email_pattern, email)
        
        return matches[0] if matches else ""
    
    def _call_claude_api(self, base64_image: str) -> Dict:
        """Call Claude API with simplified prompt and strict JSON output."""
        try:
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0,
                system="""You are a business card information extractor. Return ONLY a JSON object in the exact format shown below.

    Rules:
    1. Return ONLY JSON - no other text
    2. Extract ONLY text you can read with 100% certainty
    3. Use empty strings for unclear text
    4. Never guess or generate information
    5. Copy text exactly as shown

    JSON Format:
    {
        "name": {
            "original": "exactly as shown on card",
            "english": "english version if shown"
        },
        "title": "exact job title if shown",
        "company": {
            "original": "exactly as shown on card",
            "english": "english version if shown"
        },
        "address": {
            "office": "exact office address if shown",
            "factory": "exact factory address if shown"
        },
        "contact": {
            "mobile": "exact mobile number if shown",
            "office_phone": "exact office number if shown",
            "fax": "exact fax if shown",
            "email": "exact email if shown",
            "website": "exact website if shown"
        },
        "expo": "exact expo/booth info if shown",
        "additional_notes": "any other visible text"
    }""",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract the business card information and return ONLY a JSON object. Do not include any other text."
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
            
            # Remove any non-JSON text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            
            if json_start == -1 or json_end == -1:
                return {"error": "No JSON object found in response"}
                
            json_text = response_text[json_start:json_end + 1]
            
            # Basic JSON validation and cleaning
            try:
                data = json.loads(json_text)
                
                # Quick check for obviously fake data
                fake_data = ["JOHN DOE", "JANE DOE", "ACME", "123 MAIN", "(123)", "example.com"]
                json_str = json.dumps(data).lower()
                
                for fake in fake_data:
                    if fake.lower() in json_str:
                        return {
                            "error": "Detected generated data",
                            "details": f"Found '{fake}' in response"
                        }
                
                return data
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Problematic JSON: {json_text}")
                return {"error": f"JSON parsing error: {str(e)}"}
                
        except Exception as e:
            print(f"API call error: {str(e)}")
            return {"error": f"API call failed: {str(e)}"}
        
    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON from Claude's response with improved error handling."""
        try:
            # Remove any non-JSON text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end <= 0:
                print("No JSON object found in response")
                return {"error": "No JSON object found in response"}
                
            json_text = response_text[json_start:json_end]
            
            # Clean up the JSON text
            json_text = re.sub(r'[\n\r\t]', ' ', json_text)  # Remove newlines and tabs
            json_text = re.sub(r'\s+', ' ', json_text)  # Normalize whitespace
            json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
            
            # Fix common JSON formatting issues
            json_text = re.sub(r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', json_text)  # Quote unquoted keys
            json_text = json_text.replace("'", '"')  # Replace single quotes with double quotes
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Problematic JSON: {json_text}")
                return {"error": f"JSON parsing error: {str(e)}"}
                
            # Validate and clean the parsed data
            return self._validate_extraction(data)
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return {"error": f"Error parsing response: {str(e)}"}

    def process_single_image(self, image_path: str) -> Dict:
        """Process a single image with improved error handling and validation."""
        try:
            print(f"\nProcessing image: {image_path}")
            
            # Read and preprocess image
            with open(image_path, 'rb') as file:
                img_array = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
            if image is None:
                raise ValueError("Failed to read image")
                
            # Enhance image quality
            enhanced = self._enhance_image(image)
            
            # Detect and extract business card
            card_image = self.detect_business_card_simple(enhanced)
            
            # Save debug image
            debug_dir = Path("debug_images")
            debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_dir / f"processed_{Path(image_path).name}"), card_image)
            
            # Convert to base64
            base64_image = self.encode_image(card_image)
            
            # Call API and validate results
            result = self._call_claude_api(base64_image)
            
            # Validate and clean extracted data
            if 'error' not in result:
                result = self._validate_extraction(result)
                
            return result
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {"error": str(e)}
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better text recognition."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels
            lab = cv2.merge((l,a,b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Denoise
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced)
            
            return enhanced
            
        except Exception as e:
            print(f"Error enhancing image: {str(e)}")
            return image

    def _validate_extraction(self, data: Dict) -> Dict:
        """Validate extracted data with basic cleaning."""
        template = {
            "name": {"original": "", "english": ""},
            "title": "",
            "company": {"original": "", "english": ""},
            "address": {"office": "", "factory": ""},
            "contact": {
                "mobile": "",
                "office_phone": "",
                "fax": "",
                "email": "",
                "website": ""
            },
            "expo": "",
            "additional_notes": ""
        }
        
        if "error" in data:
            return data
            
        try:
            result = template.copy()
            
            # Basic cleaning only - no format changes
            if isinstance(data.get('name'), dict):
                result['name']['original'] = str(data['name'].get('original', '')).strip()
                result['name']['english'] = str(data['name'].get('english', '')).strip()
                
            if isinstance(data.get('company'), dict):
                result['company']['original'] = str(data['company'].get('original', '')).strip()
                result['company']['english'] = str(data['company'].get('english', '')).strip()
                
            if isinstance(data.get('address'), dict):
                result['address']['office'] = str(data['address'].get('office', '')).strip()
                result['address']['factory'] = str(data['address'].get('factory', '')).strip()
                
            if isinstance(data.get('contact'), dict):
                contact = data['contact']
                result['contact']['mobile'] = str(contact.get('mobile', '')).strip()
                result['contact']['office_phone'] = str(contact.get('office_phone', '')).strip()
                result['contact']['fax'] = str(contact.get('fax', '')).strip()
                result['contact']['email'] = str(contact.get('email', '')).strip()
                result['contact']['website'] = str(contact.get('website', '')).strip()
                
            result['title'] = str(data.get('title', '')).strip()
            result['expo'] = str(data.get('expo', '')).strip()
            result['additional_notes'] = str(data.get('additional_notes', '')).strip()
            
            return result
            
        except Exception as e:
            print(f"Error in validation: {str(e)}")
            return {"error": f"Validation failed: {str(e)}"}
    def _is_likely_fake(self, text: str) -> bool:
        """Check if text appears to be generated."""
        fake_patterns = [
            r'JOHN|JANE DOE',
            r'ACME',
            r'^\d{3} MAIN ST',
            r'EXAMPLE[.\w]+',
            r'\(123\) \d{3}-\d{4}',
            r'\b[A-Z]+ CORP\b'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) 
                for pattern in fake_patterns)
    
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