from flask import Flask
from routes.vehicle import vehicle_bp
from routes.safety import safety_bp
from routes.image import image_bp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_app():
    """
    Create and configure the Flask application, register all blueprints.
    """
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.register_blueprint(vehicle_bp)
    app.register_blueprint(safety_bp)
    app.register_blueprint(image_bp)
    return app


if __name__ == '__main__':
    app = create_app()
    # You can add production-ready config or logging here if needed
    app.run(host='0.0.0.0', port=5000, debug=False)
