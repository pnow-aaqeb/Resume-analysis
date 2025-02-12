import json

personal_information_schema = {
    "full_name": "",
    "email": "",
    "linkedin": "",
    "github": "",
    "phone": "",
    "address_string": "",
    "country_of_residence": "",
    "city_of_residence": "",
    "zip": "",
    "state_of_residence": "",
    "age": "",
    "is_us_citizen": False
}

education_schema = {
    "primary": "",
    "secondary": "",
    "graduation": "",
    "post_graduation": "",
    "phd": ""
}

location_schema = {
    "country": "",
    "short_code": "",
    "state": "",
    "city": "",
    "zip": ""
}

certifications_schema = [
    {
        "name_of_certification": "",
        "institution": ""
    }
]

work_experience_schema = [
    {
        "job_role_name": "",
        "organization_name": "",
        "is_present": False,
        "description": "",
        "start_month_year": "",
        "end_month_year": ""
    }
]

generate_data_schema = {
    "is_remote": False,
    "total_work_experience": "",
    "keywords": [],
    "technical_skills": [],
    "general_skills": [],
    "licenses": [],
    "industries": [],
    "possible_job_roles": [],
    "summary": "",
    "search_location": "",
    "resume_text":""
}

improved_resume_extraction_prompt = """
RESUME EXTRACTION AND ANALYSIS GUIDELINES...
CORE OBJECTIVES:
- The response should stricly be in JSON only.
- Extract precise, structured information from resumes
- Handle complex, non-conventional resume layouts
- Provide comprehensive structured response
EXTRACTION STRATEGY:
1. TEXT RECONSTRUCTION
- If text appears fragmented, logically reconstruct sections
- Prioritize information integrity over strict formatting
- Use contextual understanding to parse non-standard layouts
2. INFORMATION HIERARCHY
Parse resume following this priority:
a) Personal Information
b) Education
c) Work Experience
d) Skills and Certifications
e) Additional Relevant Data
3. DATA VALIDATION RULES
- Personal Information:
  * Validate email format
  * Standardize phone number
  * Confirm location details
  * Generate missing data only if absolutely necessary
- Education:
  * Extract full institution names
  * Capture degree types and specializations
  * Note graduation years
- Work Experience:
  * Capture exact job titles
  * Extract organization details
  * Preserve job descriptions verbatim
  * Determine employment duration accurately
- Skills:
  * Categorize skills (technical, soft, domain-specific)
  * Include synonyms and variations
  * Capture skill proficiency levels if mentioned
EXTRACTION GUIDELINES:
- Be extremely precise in information extraction
- Do not fabricate or assume information
- If information is unclear or missing, indicate as such
- Maintain the original context and wording
RESPONSE FORMAT:
- Each prompt has its own response.
- Strictly, do not add any comments, explanation sentences and so on, the response should be in the described structure only.
- Provide a clean, well-structured output
- Ensure all keys are populated
- Use appropriate data types
- Include empty arrays/null where no data exists
ADVANCED HANDLING:
- Handle multiple languages
- Process resumes with complex formatting
- Extract information from partially readable documents
- Minimize false positives
ERROR HANDLING:
- If extraction is impossible, return a detailed error explanation
- Provide percentage of successful extraction
- Suggest potential reasons for incomplete parsing
CONFIDENTIALITY:
- Treat all extracted information as strictly confidential
- Do not store or reproduce personal information
- Focus solely on structured data extraction
"""

prompt_personal_info_education_location_certifications = f"""
 Please convert the following resume resume in JSON only, follow these guidlines . You need to extract 'personal_information', 'education', 'location', and 'certifications' (Array). 
  You must only use the information in the resume document provided. Do not make assumptions or use external knowledge. Do not fill in assumed data and only for the zip code you can use the address and fill the zip if and only if it is not provided in the resume.
  
 personal_information: {json.dumps(personal_information_schema, indent=2)}

education: {json.dumps(education_schema, indent=2)}

location: {json.dumps(location_schema, indent=2)}

certifications: {json.dumps(certifications_schema, indent=2)}
  
  TASK:
  - In this schema, 'is_us_citizen' is a boolean, while the rest of the fields are strings. For the zip code, if not provided by the candidate, it should be generated based on the 'address_string'. The 'address_string' should contain all location details.
  - If no location data is found, generate dummy location details based on city/state/country. 
  - Ensure 'zipcode' is not missing and the 'short_code' is also included.
  
  Strictly follow the information provided in the resume and do not add external knowledge or any assumptions
"""

prompt_work_experience = f"""
 Please convert the following resume resume in JSON only. You need to extract the work/professional experience (Array).
  You must only use the information in the resume document provided. Do not make assumptions or use external knowledge.
  
 work_experience: {json.dumps(work_experience_schema, indent=2)}
  
  TASK:
  - 'is_present' is a boolean, and 'description' is the exact job description as it appears in the resume. 
  - The 'start_month_year' and 'end_month_year' should follow the format (DD/MM/YYYY to DD/MM/YYYY). 
  - If the candidate is still working, set 'end_month_year' to "present". Do not summarize the job description; it must be exactly as it appears.
  - If no work experience is found, keep the array empty. Do not assign a null value.
"""

prompt_generate_data = f"""
Please convert the following resume resume in JSON only. You need to extract 'generateData' (Object).
  You must only use the information in the resume document provided. Do not make assumptions or use external knowledge.
  
  generateData: {json.dumps(generate_data_schema, indent=2)}
  
  TASK:
  - For 'technical_skills', include all possible synonyms.
  - For 'keywords', there must be a minimum of 200 keywords strictly from the resume. Include both full forms and abbreviations. 
  - For multi-word keywords, include both the complete phrase and the individual words if they make sense.
  - For 'summary', provide a summary of the resume in 50-100 words.
  - For 'possible_job_roles', list all potential job roles this resume might qualify for.
  - Set 'is_remote' to true if the candidate has worked remotely before, otherwise set it to false.
  - 'total_work_experience' should be the total work experience in years, as a string. If no work experience is found, set it to "0".
  
  Strictly follow the information provided in the resume and do not add external knowledge or any assumptions
"""
