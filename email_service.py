import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "signsight8@gmail.com"
APP_PASSWORD = "enjp rwbe fuzy zvpv"

EMOTION_COLORS = {
    "Happy": "#22c55e",
    "Sad": "#3b82f6",
    "Angry": "#ef4444",
    "Neutral": "#64748b",
    "Fear": "#a855f7"
}

def build_html_email(report):
    rows = ""

    for r in report["results"]:
        color = EMOTION_COLORS.get(r["dominant"], "#64748b")

        distribution = "".join(
            f"""
            <tr>
                <td style="padding:6px 0;color:#374151">{e}</td>
                <td style="padding:6px 0;color:#111827">{p}%</td>
            </tr>
            """
            for e, p in r["distribution"].items() if p > 0
        )

        rows += f"""
        <div class="card">
            <h3>{r['video']}</h3>
            <p><strong>Expected Emotion:</strong> {r['expected']}</p>
            <p>
                <strong>Detected Emotion:</strong>
                <span class="badge" style="background:{color}">
                    {r['dominant']}
                </span>
            </p>
            <p><strong>Consistency Score:</strong> {r['consistency']}%</p>

            <table>
                <tr>
                    <th align="left">Emotion</th>
                    <th align="left">Percentage</th>
                </tr>
                {distribution}
            </table>
        </div>
        """

    verdict = (
        "The child demonstrates strong emotional recognition and expression skills."
        if report["accuracy"] >= 80 else
        "The child shows developing emotional awareness with some inconsistencies."
        if report["accuracy"] >= 60 else
        "The child may benefit from additional guided emotional learning support."
    )

    return f"""
    <html>
    <head>
        <style>
            body {{
                margin:0;
                padding:0;
                background:#f1f5f9;
                font-family:Arial, sans-serif;
            }}
            .container {{
                max-width:650px;
                margin:auto;
                background:#ffffff;
                padding:32px;
                border-radius:10px;
            }}
            h1 {{
                color:#0f172a;
                margin-bottom:8px;
            }}
            .subtitle {{
                color:#475569;
                margin-bottom:24px;
            }}
            .card {{
                background:#f8fafc;
                border-radius:10px;
                padding:20px;
                margin-bottom:20px;
            }}
            h3 {{
                margin-top:0;
                color:#020617;
            }}
            table {{
                width:100%;
                border-collapse:collapse;
                margin-top:12px;
            }}
            th {{
                color:#475569;
                font-size:14px;
                padding-bottom:6px;
            }}
            .badge {{
                padding:6px 12px;
                border-radius:999px;
                color:#fff;
                font-size:13px;
                font-weight:bold;
            }}
            .summary {{
                background:#eef2ff;
                padding:20px;
                border-radius:10px;
                margin-top:24px;
            }}
            .footer {{
                margin-top:30px;
                font-size:13px;
                color:#64748b;
                text-align:center;
            }}
        </style>
    </head>

    <body>
        <div class="container">
            <h1>Emotional Development Assessment</h1>
            <p class="subtitle">
                Child Assessment Date: {report['date']} <br/>
                Overall Accuracy: <strong>{report['accuracy']}%</strong>
            </p>

            {rows}

            <div class="summary">
                <h3>Professional Interpretation</h3>
                <p>{verdict}</p>

                <h3>Recommendations</h3>
                <ul>
                    <li>Continue structured emotion practice sessions</li>
                    <li>Use mirrors and visual emotion cards</li>
                    <li>Encourage safe and open emotional expression</li>
                    <li>Review recordings together for positive reinforcement</li>
                </ul>
            </div>

            <div class="footer">
                This report was generated automatically by SignSight<br/>
                Supporting emotional development through technology
            </div>
        </div>
    </body>
    </html>
    """

def send_emotion_report_email(receiver, report):
    msg = MIMEMultipart("alternative")
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver
    msg["Subject"] = "Child Emotional Expression Assessment Report"

    html_content = build_html_email(report)
    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)