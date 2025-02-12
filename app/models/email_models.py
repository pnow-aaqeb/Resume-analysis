from pydantic import BaseModel
from typing import Optional, List

class EmailContent(BaseModel):
    subject: Optional[str]
    body: str
    sender: Optional[str]
    receiver: Optional[str]

class Attachment(BaseModel):
    file_url: str
    file_name: Optional[str] = None

class EmailRequest(BaseModel):
    email_content: EmailContent
    attachments: List[Attachment] = []