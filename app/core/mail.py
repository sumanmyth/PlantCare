from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from app.core.config import settings

from typing import List

conf = ConnectionConfig(
    MAIL_USERNAME=settings.smtp_username,
    MAIL_PASSWORD=settings.smtp_password,
    MAIL_FROM=settings.email_from,
    MAIL_PORT=settings.smtp_port,
    MAIL_SERVER=settings.smtp_server,
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
)

fm = FastMail(conf)

async def send_email(subject: str, recipient_email: List[str], body: str):
    message = MessageSchema(
        subject=subject,
        recipients=recipient_email,
        body=body,
        subtype="html",
    )
    await fm.send_message(message)
