"""
quantum_models/

Quantum circuit integration variants of the TF-CLIP model.
Each module targets a distinct integration category from the paper:

    make_model_qclassifier.py  -- Category: Classifier
                                   VQC replaces the four nn.Linear classifier heads.

Shared quantum building blocks:
    quantum_layers.py          -- QuantumClassifier and other reusable QNN modules.

Usage (from project root):
    from quantum_models.make_model_qclassifier import make_model
    model = make_model(cfg, num_class, camera_num, view_num,
                       n_qubits=8, n_layers=2)
"""
