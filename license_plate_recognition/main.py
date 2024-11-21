from .recognition.service import RecognitionService

print("Polish license plate recognition system")

recognition_service = RecognitionService()
recognition_service.recognize()
