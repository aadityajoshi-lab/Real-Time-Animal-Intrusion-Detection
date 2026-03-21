# FarmGuard AI - Real-Time Animal Intrusion Detection

Live camera-based animal intrusion detection system using deep learning for farm protection.

## Features

- Real-time animal detection using RT-DETR
- Multi-camera stream support
- Telegram alerts with detection images
- Repellent sound control system
- False alarm feedback mechanism
- User authentication and management

## Tech Stack

**Backend:**
- Django 5.x (Web framework)
- FastAPI (Detection API)
- RT-DETR (Object detection)
- OpenCV (Video processing)

**Frontend:**
- HTML/CSS/JavaScript
- Responsive design

**Deployment:**
- Modal.com (Serverless GPU inference)
- SQLite/PostgreSQL (Database)

## Project Structure

```
detection/
├── backend_fastapi/       # FastAPI detection service
│   ├── main.py           # Main API endpoints
│   ├── telegram_bot.py   # Telegram integration
│   └── alert_agent.py    # Alert dispatch system
├── core/                  # Django app
│   ├── models.py         # Database models
│   ├── views.py          # View controllers
│   └── api_proxy.py      # API proxy to FastAPI
├── detection/             # Django project settings
├── templates/             # HTML templates
├── static/               # Static assets
└── modal_web.py          # Modal deployment config
```

## Setup

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r backend_fastapi/requirements.txt
   ```
4. Copy environment files:
   ```bash
   cp .env.example .env
   cp backend_fastapi/.env.example backend_fastapi/.env
   ```
5. Configure environment variables in `.env` files
6. Run migrations:
   ```bash
   python manage.py migrate
   ```
7. Start the servers:
   ```bash
   # Terminal 1 - Django
   python manage.py runserver
   
   # Terminal 2 - FastAPI
   cd backend_fastapi
   uvicorn main:app --port 8001
   ```

## Deployment

Deploy to Modal.com:
```bash
modal deploy modal_web.py
```

## Detected Animals

- Elephant
- Tiger
- Bear
- Leopard
- Wild Boar
- Nilgai
- Monkey
- Jackal
- Gaur

## License

MIT License

## Contributors

- Aaditya Joshi
- Aayush
- Aashrav
- Bikash
