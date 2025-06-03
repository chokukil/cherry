#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporary script to fix the Data_Agent.py file by removing duplicate function definitions
"""

def fix_data_agent_file():
    """Remove duplicate function definitions from the end of the file"""
    file_path = 'pages/03_Data_Agent.py'
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Original file has {len(lines)} lines")
        
        # Find the last occurrence of duplicate functions
        # Look for the last "# 🔧 DataFrame 안전 접근 헬퍼 함수 추가" comment
        last_duplicate_start = -1
        for i in range(len(lines) - 1, -1, -1):
            if "# 🔧 DataFrame 안전 접근 헬퍼 함수 추가" in lines[i]:
                last_duplicate_start = i
                break
        
        if last_duplicate_start != -1:
            print(f"Found duplicate functions starting at line {last_duplicate_start + 1}")
            # Remove from the duplicate comment onwards
            lines = lines[:last_duplicate_start]
            print(f"Trimmed to {len(lines)} lines")
        else:
            print("No duplicate function definitions found")
            return False
        
        # Write the fixed file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("File fixed successfully!")
        return True
        
    except Exception as e:
        print(f"Error fixing file: {e}")
        return False

if __name__ == "__main__":
    fix_data_agent_file() 