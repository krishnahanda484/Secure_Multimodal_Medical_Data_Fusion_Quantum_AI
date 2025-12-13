"""
Quantum Security Module
Implements quantum-safe encryption and quantum-enhanced signatures
"""

import torch
import numpy as np
import hashlib
import os
import logging
from cryptography.fernet import Fernet
import pennylane as qml

from project_config import (
    MODELS_DIR,
    SECURITY_KEY_PATH,
    ENCRYPTED_MODEL_PATH,
    N_QUBITS,
    QUANTUM_BACKEND
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# QUANTUM SIGNATURE GENERATOR
# ============================================
class QuantumSignature:
    """
    Generates quantum-enhanced signatures for data integrity
    Uses quantum circuits to create unique signatures
    """
    
    def __init__(self, n_qubits=N_QUBITS):
        self.n_qubits = n_qubits
        self.dev = qml.device(QUANTUM_BACKEND, wires=n_qubits)
        
        # Define quantum circuit
        @qml.qnode(self.dev)
        def quantum_circuit(inputs, weights):
            # Angle embedding - encode classical data into quantum states
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            
            # Entangling layer - creates quantum correlations
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            
            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.circuit = quantum_circuit
    
    def generate_signature(self, data_string, confidence_score=None):
        """
        Generate quantum signature for given data
        
        Args:
            data_string: String to sign (e.g., diagnosis result)
            confidence_score: Optional confidence score to include
            
        Returns:
            str: Hexadecimal quantum signature
        """
        # Combine data
        if confidence_score is not None:
            combined_data = f"{data_string}-{confidence_score}"
        else:
            combined_data = data_string
        
        # Create deterministic seed from data
        hash_obj = hashlib.sha256(combined_data.encode('utf-8'))
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        
        # Generate quantum parameters
        np.random.seed(seed)
        inputs = np.random.uniform(0, 2*np.pi, self.n_qubits)
        weights = np.random.uniform(0, 2*np.pi, (1, self.n_qubits))
        
        # Execute quantum circuit
        quantum_measurements = self.circuit(inputs, weights)
        
        # Convert quantum measurements to signature
        signature_bytes = np.array(quantum_measurements).tobytes()
        quantum_signature = hashlib.sha256(signature_bytes).hexdigest()
        
        return quantum_signature
    
    def verify_signature(self, data_string, signature, confidence_score=None):
        """
        Verify a quantum signature
        
        Returns:
            bool: True if signature is valid
        """
        regenerated_sig = self.generate_signature(data_string, confidence_score)
        return regenerated_sig == signature


# ============================================
# QUANTUM-SAFE ENCRYPTION
# ============================================
class QuantumSafeEncryption:
    """
    Implements AES-256 encryption (quantum-safe for now)
    In future, can be replaced with post-quantum cryptography (e.g., CRYSTALS-Kyber)
    """
    
    def __init__(self):
        self.key = None
        self.fernet = None
    
    def generate_key(self):
        """Generate new encryption key"""
        self.key = Fernet.generate_key()
        self.fernet = Fernet(self.key)
        return self.key
    
    def load_key(self, key_path):
        """Load existing key"""
        with open(key_path, 'rb') as f:
            self.key = f.read()
        self.fernet = Fernet(self.key)
    
    def save_key(self, key_path):
        """Save encryption key"""
        if self.key is None:
            raise ValueError("No key to save. Generate or load a key first.")
        with open(key_path, 'wb') as f:
            f.write(self.key)
        logger.info(f"Encryption key saved to {key_path}")
    
    def encrypt_data(self, data):
        """Encrypt data"""
        if self.fernet is None:
            raise ValueError("No encryption key loaded.")
        return self.fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data):
        """Decrypt data"""
        if self.fernet is None:
            raise ValueError("No decryption key loaded.")
        return self.fernet.decrypt(encrypted_data)


# ============================================
# MODEL ENCRYPTION
# ============================================
class ModelSecurity:
    """
    Handles encryption and decryption of trained models
    """
    
    def __init__(self):
        self.encryptor = QuantumSafeEncryption()
    
    def encrypt_model(self, model_path, output_path, key_path):
        """
        Encrypt a PyTorch model
        
        Args:
            model_path: Path to model state dict
            output_path: Path to save encrypted model
            key_path: Path to save encryption key
        """
        logger.info(f"Encrypting model: {model_path}")
        
        # Generate encryption key
        self.encryptor.generate_key()
        self.encryptor.save_key(key_path)
        
        # Read model
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        # Encrypt
        encrypted_data = self.encryptor.encrypt_data(model_data)
        
        # Save encrypted model
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        logger.info(f"✅ Model encrypted and saved to {output_path}")
    
    def decrypt_model(self, encrypted_path, key_path):
        """
        Decrypt a PyTorch model and return state dict
        
        Args:
            encrypted_path: Path to encrypted model
            key_path: Path to encryption key
            
        Returns:
            dict: Model state dict
        """
        logger.info(f"Decrypting model: {encrypted_path}")
        
        # Load encryption key
        self.encryptor.load_key(key_path)
        
        # Read encrypted model
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        # Decrypt
        decrypted_data = self.encryptor.decrypt_data(encrypted_data)
        
        # Load state dict
        import io
        state_dict = torch.load(io.BytesIO(decrypted_data))
        
        logger.info("✅ Model decrypted successfully")
        return state_dict
    
    def encrypt_all_models(self):
        """Encrypt all models in the models directory"""
        logger.info("=" * 50)
        logger.info("Encrypting All Models")
        logger.info("=" * 50)
        
        models_to_encrypt = [
            'best_fusion.pth',
            'best_classifier.pth',
            'final_fusion.pth',
            'final_classifier.pth'
        ]
        
        for model_name in models_to_encrypt:
            model_path = os.path.join(MODELS_DIR, model_name)
            
            if os.path.exists(model_path):
                encrypted_path = os.path.join(MODELS_DIR, f"{model_name}.enc")
                key_path = os.path.join(MODELS_DIR, f"{model_name}.key")
                
                self.encrypt_model(model_path, encrypted_path, key_path)
            else:
                logger.warning(f"Model not found: {model_path}")
        
        logger.info("=" * 50)
        logger.info("Encryption Complete!")
        logger.info("=" * 50)


# ============================================
# DATA ANONYMIZATION
# ============================================
class DataAnonymizer:
    """
    Anonymizes patient data before processing
    """
    
    @staticmethod
    def hash_patient_id(patient_id):
        """Create hashed patient ID"""
        return hashlib.sha256(str(patient_id).encode()).hexdigest()[:16]
    
    @staticmethod
    def anonymize_report(report_text):
        """
        Remove or hash personally identifiable information from reports
        This is a simplified version - production would need more sophisticated NER
        """
        # Common PII patterns (simplified)
        import re
        
        # Remove common name patterns
        anonymized = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', report_text)
        
        # Remove dates
        anonymized = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '[DATE]', anonymized)
        
        # Remove phone numbers
        anonymized = re.sub(r'\d{3}[-.]?\d{3}[-.]?\d{4}', '[PHONE]', anonymized)
        
        # Remove SSN-like patterns
        anonymized = re.sub(r'\d{3}-\d{2}-\d{4}', '[SSN]', anonymized)
        
        return anonymized


# ============================================
# MAIN FUNCTIONS
# ============================================
def demo_quantum_signature():
    """Demonstrate quantum signature generation"""
    logger.info("\n" + "=" * 50)
    logger.info("Quantum Signature Demo")
    logger.info("=" * 50)
    
    qsig = QuantumSignature()
    
    # Generate signatures for different diagnoses
    test_cases = [
        ("Brain: Tumor Detected", 0.95),
        ("Lung: Pneumonia", 0.87),
        ("Bone: Fracture", 0.92)
    ]
    
    for diagnosis, confidence in test_cases:
        signature = qsig.generate_signature(diagnosis, confidence)
        logger.info(f"\nDiagnosis: {diagnosis}")
        logger.info(f"Confidence: {confidence}")
        logger.info(f"Quantum Signature: {signature}")
        
        # Verify signature
        is_valid = qsig.verify_signature(diagnosis, signature, confidence)
        logger.info(f"Signature Valid: {is_valid}")


def demo_encryption():
    """Demonstrate model encryption"""
    logger.info("\n" + "=" * 50)
    logger.info("Model Encryption Demo")
    logger.info("=" * 50)
    
    security = ModelSecurity()
    security.encrypt_all_models()


def main():
    """Main execution"""
    # Demo quantum signatures
    demo_quantum_signature()
    
    # Demo encryption (if models exist)
    demo_encryption()


if __name__ == "__main__":
    main()