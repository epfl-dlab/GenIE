# TODO: remove all references to this file and replace with hydra params
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["HYDRA_FULL_ERROR"] = "1"

# directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SURFACE_FORM_DICTS = os.path.join(DATA_DIR, "surface_form_dicts")
SURFACE_FORMS_FROM_SNAPSHOT = os.path.join(DATA_DIR, "surface_form_dicts_from_snapshot")
# assert os.path.isdir(BASE_DIR) and os.path.isdir(DATA_DIR)
