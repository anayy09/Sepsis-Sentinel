"""
Flask Dashboard for Sepsis Sentinel
Real-time monitoring dashboard with live risk scores and model explanations.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import requests
from flask import Flask, render_template, jsonify, request, session
from flask_socketio import SocketIO, emit, join_room, leave_room
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sepsis-sentinel-dashboard-secret-key'
app.config['JSON_SORT_KEYS'] = False

# SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state management
thread_lock = Lock()
background_thread = None
connected_clients = {}
patient_data = {}  # Cache for patient data
alert_history = []  # Recent alerts


class DashboardState:
    """Manages dashboard state and real-time data."""
    
    def __init__(self):
        self.active_patients = {}
        self.risk_history = {}
        self.alerts = []
        self.system_metrics = {
            'total_predictions': 0,
            'high_risk_patients': 0,
            'avg_processing_time': 0.0,
            'model_accuracy': 0.92
        }
        self.last_update = datetime.now()
    
    def add_patient_data(self, patient_id: str, data: dict):
        """Add or update patient data."""
        self.active_patients[patient_id] = {
            **data,
            'last_updated': datetime.now()
        }
        
        # Update risk history
        if patient_id not in self.risk_history:
            self.risk_history[patient_id] = []
        
        self.risk_history[patient_id].append({
            'timestamp': datetime.now(),
            'risk_score': data.get('sepsis_probability', 0.0),
            'risk_level': data.get('sepsis_risk_level', 'low')
        })
        
        # Keep only last 100 points
        if len(self.risk_history[patient_id]) > 100:
            self.risk_history[patient_id] = self.risk_history[patient_id][-100:]
    
    def add_alert(self, alert_data: dict):
        """Add new alert."""
        alert_data['timestamp'] = datetime.now()
        self.alerts.append(alert_data)
        
        # Keep only last 50 alerts
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
    
    def get_summary_stats(self) -> dict:
        """Get dashboard summary statistics."""
        total_patients = len(self.active_patients)
        high_risk_count = sum(
            1 for p in self.active_patients.values()
            if p.get('sepsis_risk_level') in ['high', 'critical']
        )
        
        avg_risk = np.mean([
            p.get('sepsis_probability', 0.0)
            for p in self.active_patients.values()
        ]) if self.active_patients else 0.0
        
        return {
            'total_patients': total_patients,
            'high_risk_patients': high_risk_count,
            'average_risk': avg_risk,
            'active_alerts': len([a for a in self.alerts if 
                               (datetime.now() - a['timestamp']).seconds < 3600])
        }


# Global dashboard state
dashboard_state = DashboardState()


def generate_mock_patient_data() -> dict:
    """Generate mock patient data for demonstration."""
    import random
    
    patient_id = f"PAT_{random.randint(1000, 9999)}"
    
    # Generate realistic sepsis probability
    base_risk = random.uniform(0.05, 0.95)
    
    # Add some temporal correlation
    if patient_id in dashboard_state.risk_history:
        last_risk = dashboard_state.risk_history[patient_id][-1]['risk_score']
        base_risk = max(0.0, min(1.0, last_risk + random.gauss(0, 0.1)))
    
    risk_level = determine_risk_level(base_risk)
    
    # Generate mock vital signs
    vitals = {
        'heart_rate': random.gauss(80, 15) + (base_risk * 40),
        'systolic_bp': random.gauss(120, 20) - (base_risk * 30),
        'diastolic_bp': random.gauss(80, 10) - (base_risk * 20),
        'temperature': random.gauss(98.6, 2) + (base_risk * 4),
        'respiratory_rate': random.gauss(16, 4) + (base_risk * 8),
        'spo2': random.gauss(98, 3) - (base_risk * 8)
    }
    
    # Generate mock demographics
    demographics = {
        'age': random.randint(25, 85),
        'gender': random.choice(['M', 'F']),
        'admission_type': random.choice(['Emergency', 'Elective', 'Urgent'])
    }
    
    return {
        'patient_id': patient_id,
        'sepsis_probability': base_risk,
        'sepsis_risk_level': risk_level,
        'confidence_score': random.uniform(0.7, 0.95),
        'vitals': vitals,
        'demographics': demographics,
        'processing_time_ms': random.uniform(50, 200),
        'timestamp': datetime.now()
    }


def determine_risk_level(probability: float) -> str:
    """Determine risk level from probability."""
    if probability < 0.2:
        return "low"
    elif probability < 0.5:
        return "medium" 
    elif probability < 0.8:
        return "high"
    else:
        return "critical"


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level."""
    colors = {
        'low': '#28a745',
        'medium': '#ffc107', 
        'high': '#fd7e14',
        'critical': '#dc3545'
    }
    return colors.get(risk_level, '#6c757d')


def create_risk_gauge(risk_score: float, patient_id: str) -> dict:
    """Create Plotly gauge chart for risk score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        title={'text': f"Sepsis Risk - {patient_id}"},
        delta={'reference': 20, 'position': "top"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "#d4edda"},
                {'range': [20, 50], 'color': "#fff3cd"},
                {'range': [50, 80], 'color': "#f8d7da"},
                {'range': [80, 100], 'color': "#f5c6cb"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white"
    )
    
    return plotly.utils.PlotlyJSONEncoder().encode(fig)


def create_risk_timeline(patient_id: str) -> dict:
    """Create timeline chart for patient risk history."""
    if patient_id not in dashboard_state.risk_history:
        return {}
    
    history = dashboard_state.risk_history[patient_id]
    
    if not history:
        return {}
    
    timestamps = [h['timestamp'] for h in history]
    risk_scores = [h['risk_score'] * 100 for h in history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=risk_scores,
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#007bff', width=2),
        marker=dict(size=6)
    ))
    
    # Add risk threshold lines
    fig.add_hline(y=20, line_dash="dash", line_color="green", 
                  annotation_text="Low Risk")
    fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Risk")
    fig.add_hline(y=80, line_dash="dash", line_color="red", 
                  annotation_text="High Risk")
    
    fig.update_layout(
        title=f"Risk Timeline - {patient_id}",
        xaxis_title="Time",
        yaxis_title="Risk Score (%)",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="white"
    )
    
    return plotly.utils.PlotlyJSONEncoder().encode(fig)


def create_shap_waterfall(patient_id: str) -> dict:
    """Create mock SHAP waterfall chart."""
    # Mock SHAP values for demonstration
    features = [
        'Heart Rate', 'Temperature', 'WBC Count', 'Lactate',
        'Blood Pressure', 'Respiratory Rate', 'Age', 'GCS Score'
    ]
    
    shap_values = np.random.normal(0, 0.1, len(features))
    base_value = 0.15
    
    # Sort by absolute SHAP value
    sorted_indices = np.argsort(np.abs(shap_values))[::-1]
    
    fig = go.Figure()
    
    cumulative = base_value
    x_pos = list(range(len(features) + 2))
    
    # Base value
    fig.add_trace(go.Bar(
        x=[0], y=[base_value],
        name='Base Value',
        marker_color='gray',
        text=[f'{base_value:.3f}'],
        textposition='auto'
    ))
    
    # Feature contributions
    for i, idx in enumerate(sorted_indices):
        fig.add_trace(go.Bar(
            x=[i + 1], y=[shap_values[idx]],
            name=features[idx],
            marker_color='red' if shap_values[idx] > 0 else 'blue',
            text=[f'{shap_values[idx]:.3f}'],
            textposition='auto'
        ))
        cumulative += shap_values[idx]
    
    # Final prediction
    fig.add_trace(go.Bar(
        x=[len(features) + 1], y=[cumulative],
        name='Final Prediction',
        marker_color='black',
        text=[f'{cumulative:.3f}'],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"SHAP Explanation - {patient_id}",
        xaxis_title="Features",
        yaxis_title="Contribution to Risk",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="white",
        showlegend=False
    )
    
    return plotly.utils.PlotlyJSONEncoder().encode(fig)


@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/patient/<patient_id>')
def patient_detail(patient_id):
    """Patient detail page."""
    if patient_id not in dashboard_state.active_patients:
        return "Patient not found", 404
    
    patient_data = dashboard_state.active_patients[patient_id]
    return render_template('patient_detail.html', 
                         patient=patient_data, 
                         patient_id=patient_id)


@app.route('/api/dashboard/summary')
def api_summary():
    """API endpoint for dashboard summary."""
    stats = dashboard_state.get_summary_stats()
    
    return jsonify({
        'stats': stats,
        'last_update': dashboard_state.last_update.isoformat(),
        'system_status': 'healthy'
    })


@app.route('/api/patients')
def api_patients():
    """API endpoint for active patients."""
    patients = []
    
    for patient_id, data in dashboard_state.active_patients.items():
        patients.append({
            'patient_id': patient_id,
            'sepsis_probability': data.get('sepsis_probability', 0.0),
            'sepsis_risk_level': data.get('sepsis_risk_level', 'low'),
            'last_updated': data.get('last_updated', datetime.now()).isoformat(),
            'vitals': data.get('vitals', {}),
            'demographics': data.get('demographics', {})
        })
    
    # Sort by risk score descending
    patients.sort(key=lambda x: x['sepsis_probability'], reverse=True)
    
    return jsonify({'patients': patients})


@app.route('/api/patient/<patient_id>/chart/<chart_type>')
def api_patient_chart(patient_id, chart_type):
    """API endpoint for patient charts."""
    if patient_id not in dashboard_state.active_patients:
        return jsonify({'error': 'Patient not found'}), 404
    
    patient_data = dashboard_state.active_patients[patient_id]
    
    if chart_type == 'gauge':
        chart_json = create_risk_gauge(
            patient_data.get('sepsis_probability', 0.0),
            patient_id
        )
    elif chart_type == 'timeline':
        chart_json = create_risk_timeline(patient_id)
    elif chart_type == 'shap':
        chart_json = create_shap_waterfall(patient_id)
    else:
        return jsonify({'error': 'Invalid chart type'}), 400
    
    return jsonify({'chart': chart_json})


@app.route('/api/alerts')
def api_alerts():
    """API endpoint for recent alerts."""
    recent_alerts = [
        {
            **alert,
            'timestamp': alert['timestamp'].isoformat()
        }
        for alert in dashboard_state.alerts[-10:]  # Last 10 alerts
    ]
    
    return jsonify({'alerts': recent_alerts})


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    client_id = request.sid
    connected_clients[client_id] = {
        'connected_at': datetime.now(),
        'room': 'dashboard'
    }
    
    join_room('dashboard')
    
    # Send initial data
    emit('dashboard_update', {
        'type': 'initial',
        'data': dashboard_state.get_summary_stats()
    })
    
    logger.info(f"Client {client_id} connected to dashboard")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    client_id = request.sid
    if client_id in connected_clients:
        leave_room('dashboard')
        del connected_clients[client_id]
        logger.info(f"Client {client_id} disconnected from dashboard")


@socketio.on('subscribe_patient')
def handle_subscribe_patient(data):
    """Handle patient subscription."""
    patient_id = data.get('patient_id')
    client_id = request.sid
    
    if patient_id:
        join_room(f'patient_{patient_id}')
        logger.info(f"Client {client_id} subscribed to patient {patient_id}")


@socketio.on('unsubscribe_patient')
def handle_unsubscribe_patient(data):
    """Handle patient unsubscription."""
    patient_id = data.get('patient_id')
    client_id = request.sid
    
    if patient_id:
        leave_room(f'patient_{patient_id}')
        logger.info(f"Client {client_id} unsubscribed from patient {patient_id}")


def background_data_generator():
    """Background thread to generate mock data and send updates."""
    while True:
        try:
            # Generate mock patient data
            for _ in range(3):  # Add/update 3 patients per cycle
                patient_data = generate_mock_patient_data()
                dashboard_state.add_patient_data(
                    patient_data['patient_id'], 
                    patient_data
                )
                
                # Check for alerts
                if patient_data['sepsis_risk_level'] in ['high', 'critical']:
                    alert_data = {
                        'patient_id': patient_data['patient_id'],
                        'risk_level': patient_data['sepsis_risk_level'],
                        'risk_score': patient_data['sepsis_probability'],
                        'type': 'high_risk_alert'
                    }
                    dashboard_state.add_alert(alert_data)
                    
                    # Emit alert to all clients
                    socketio.emit('alert', alert_data, room='dashboard')
                
                # Emit patient update
                socketio.emit('patient_update', patient_data, 
                            room=f"patient_{patient_data['patient_id']}")
            
            # Update dashboard summary
            summary_stats = dashboard_state.get_summary_stats()
            socketio.emit('dashboard_update', {
                'type': 'summary',
                'data': summary_stats
            }, room='dashboard')
            
            dashboard_state.last_update = datetime.now()
            
            # Sleep for 5 seconds
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in background thread: {e}")
            time.sleep(10)


@socketio.on('start_background_task')
def handle_start_background():
    """Start background data generation."""
    global background_thread
    with thread_lock:
        if background_thread is None:
            background_thread = socketio.start_background_task(background_data_generator)
            logger.info("Background data generation started")


# Create templates directory if it doesn't exist
import os
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
os.makedirs(template_dir, exist_ok=True)


if __name__ == '__main__':
    # Auto-start background task
    socketio.start_background_task(background_data_generator)
    
    # Run the dashboard
    socketio.run(
        app,
        debug=False,
        host='0.0.0.0',
        port=5000,
        allow_unsafe_werkzeug=True
    )
