import os, json, logging, threading 
from flask import Flask, send_file, jsonify, request
from back_whisper import run
from value_singleton import shared_value
#start flask application
app = Flask(__name__)

#load config.json (if needed)
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
if os.path.exists(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
# Log file path is defined in config.json
log_path = config['paths']['log']
# Clear existing log file on startup
with open(log_path, 'w'):
    pass
# Configure logging format
logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#Path to the main HTML file (flask)
index_path = os.path.join(os.path.dirname(__file__), config['paths']['index'])

#variable to track listening state
listening = False

#background thread
def run_in_background():
    run('foo')  #does listening

    #- If listening is turned ON, starts Whisper in a new thread.
    #- If listening is turned OFF, sends an exit signal to Whisper. 
@app.route('/toggle_listening', methods=['POST'])
def toggle_listening():
    logger.info("Toggle Start")
    global listening
    listening = not listening
    if listening:
        logger.info("Thread Starting")
        thread = threading.Thread(target=run_in_background)
        thread.start()
        logger.info("Thread Running")
    else:
        run('exit')
        # Return the updated listening state to frontend
    return jsonify(listening=listening)

    #Returns latest transcribed text to the frontend.
    #The frontend polls (simulate real-time updates).
@app.route('/get_live_input', methods=['GET'])
def get_live_input():
    logger.info("Live Input Start")
    global listening
    if listening:
        return jsonify(input=shared_value.get_value()) 
    else:
        return jsonify(input="Listening is OFF")

    #Serves the main frontend HTML
@app.route('/')
def home():
    return send_file(index_path)
    #prevents Flask from starting multiple transcription threads
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    logger.info("Starting App")
