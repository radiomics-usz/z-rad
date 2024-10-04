import json
import logging
import os
from datetime import datetime

from joblib import Parallel, delayed

from ._base_tab import BaseTab, load_images, load_mask
from .toolbox_gui import CustomLabel, CustomBox, CustomTextField, CustomWarningBox, CustomCheckBox, \
    CustomInfo, CustomInfoBox
from ..logic.exceptions import InvalidInputParametersError, DataStructureError
from ..logic.image import get_all_structure_names, get_dicom_files
from ..logic.preprocessing import Preprocessing
from ..logic.toolbox_logic import get_logger, close_all_loggers

logging.captureWarnings(True)


def process_patient_folder(input_params, patient_folder, structure_set):
    """Function to process each patient folder, used for parallel processing."""
    # Logger
    logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logger = get_logger(logger_date_time + '_Preprocessing')

    # Initialize Preprocessing instance
    prep_image = Preprocessing(
        input_data_type=input_params["input_data_type"],
        input_imaging_modality=input_params["input_imaging_modality"],
        just_save_as_nifti=input_params["just_save_as_nifti"],
        resample_resolution=input_params["resample_resolution"],
        resample_dimension=input_params["resample_dimension"],
        interpolation_method=input_params["image_interpolation_method"],
    )
    prep_mask = Preprocessing(
        input_data_type=input_params["input_data_type"],
        input_imaging_modality=input_params["input_imaging_modality"],
        just_save_as_nifti=input_params["just_save_as_nifti"],
        resample_resolution=input_params["resample_resolution"],
        resample_dimension=input_params["resample_dimension"],
        interpolation_method=input_params["mask_interpolation_method"],
        interpolation_threshold=input_params["mask_interpolation_threshold"]
    )

    logger.info(f"Processing patient's {patient_folder} image.")
    try:
        image = load_images(input_params, patient_folder)
    except DataStructureError as e:
        logger.error(e)
        logger.error(f"Patient {patient_folder} could not be loaded and is skipped.")
        return

    if input_params["just_save_as_nifti"]:
        image_new = image.copy()
    else:
        image_new = prep_image.resample(image, image_type='image')

    # Save new image
    output_path = os.path.join(input_params["output_directory"], patient_folder, 'image.nii.gz')
    image_new.save_as_nifti(output_path)

    if input_params["use_all_structures"]:
        input_directory = os.path.join(input_params["input_directory"], patient_folder)
        rtstruct_path = get_dicom_files(input_directory, modality='RTSTRUCT')[0]
        structure_set = get_all_structure_names(rtstruct_path)

    if structure_set:
        mask_union = None
        for mask_name in structure_set:
            mask = load_mask(input_params, patient_folder, mask_name, image)
            if mask and mask.array is not None:
                logger.info(
                    f"Processing patient's {patient_folder} ROI: {mask_name}.")
                if input_params["just_save_as_nifti"]:
                    mask_new = mask.copy()
                else:
                    mask_new = prep_mask.resample(mask, image_type='mask')

                # Save new mask
                output_path = os.path.join(input_params["output_directory"], patient_folder, f'{mask_name}.nii.gz')
                mask_new.save_as_nifti(output_path)

                if input_params["mask_union"]:
                    import numpy as np
                    if mask_union:
                        mask_union.array = np.bitwise_or(mask_union.array, mask_new.array).astype(np.int16)
                    else:
                        mask_union = mask_new.copy()

        if mask_union:
            output_path = os.path.join(input_params["output_directory"], patient_folder, f'mask_union.nii.gz')
            mask_union.save_as_nifti(output_path)


class PreprocessingTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.init_dicom_elements()
        self.init_nifti_elements()
        self.init_preprocessing_elements()
        self.connect_signals()

    def init_dicom_elements(self):
        self.dicom_structures_label = CustomLabel(
            'Structures:',
            200, 300, 200, 50, self,
            style="color: white;"
        )
        self.dicom_structures_text_field = CustomTextField(
            "E.g. CTV, liver, ...",
            300, 300, 400, 50, self
        )

        self.dicom_structures_info_label = CustomInfo(
            ' i',
            'Type ROIs of interest (e.g. CTV, liver).',
            710, 300, 14, 14, self
        )

        self.use_all_structures_check_box = CustomCheckBox(
            'All structures',
            750, 300, 150, 50, self)

        self.just_save_as_nifti_check_box = CustomCheckBox(
            'Convert to NIfTI without resampling',
            900, 300, 400, 50, self)

        self._hide_dicom_elements()

    def init_nifti_elements(self):
        """Initialize UI components related to NIfTI elements."""
        # Structures
        self.nifti_structures_label = CustomLabel(
            'NIfTI Masks:',
            200, 300, 200, 50, self,
            style="color: white;"
        )
        self.nifti_structures_text_field = CustomTextField(
            "E.g. CTV, liver, ...",
            300, 300, 230, 50, self
        )
        self.nifti_structures_info_label = CustomInfo(
            ' i',
            'Provide the names of the NIfTI masks you are interested in, excluding the file extensions.'
            '\nFor example, if the files you are interested in are GTV.nii.gz and liver.nii, enter: GTV, liver.',
            545, 300, 14, 14, self
        )

        # Image
        self.nifti_image_label = CustomLabel(
            'NIfTI Image:',
            600, 300, 200, 50, self,
            style="color: white;"
        )
        self.nifti_image_text_field = CustomTextField(
            "E.g. imageCT",
            700, 300, 120, 50, self
        )
        self.nifti_image_info_label = CustomInfo(
            ' i',
            'Specify NIfTI image file without file extension',
            830, 300, 14, 14, self
        )
        self._hide_nifti_elements()

    def init_preprocessing_elements(self):
        # Resample Resolution Label and TextField
        self.resample_resolution_label = CustomLabel(
            'Resample Resolution (mm):',
            200, 380, 300, 50, self,
            style="color: white;"
        )
        self.resample_resolution_text_field = CustomTextField(
            "E.g. 1", 420, 380, 90, 50, self
        )

        # Mask union
        self.mask_union_check_box = CustomCheckBox(
            'Mask Union',
            580, 380, 150, 50, self)

        # Image Interpolation Method ComboBox
        self.image_interpolation_method_combo_box = CustomBox(
            775, 380, 210, 50, self,
            item_list=[
                'Image Interpolation:', "NN", "Linear", "BSpline", "Gaussian"
            ]
        )

        # Resample Dimension ComboBox
        self.resample_dimension_combo_box = CustomBox(
            1000, 380, 210, 50, self,
            item_list=[
                'Resample Dimension:', "2D", "3D"
            ]
        )

        # Mask Interpolation Method ComboBox
        self.mask_interpolation_method_combo_box = CustomBox(
            200, 460, 210, 50, self,
            item_list=[
                'Mask Interpolation:', "NN", "Linear", "BSpline", "Gaussian"
            ]
        )

        # Mask Interpolation Threshold Label and TextField
        self.mask_interpolation_threshold_label = CustomLabel(
            'Mask Interpolation Threshold:',
            600, 460, 360, 50, self,
            style="color: white;"
        )
        self.mask_interpolation_threshold_text_field = CustomTextField(
            "E.g. 0.75",
            830, 460, 100, 50, self
        )
        self.mask_interpolation_threshold_text_field.setText('0.5')
        self.mask_interpolation_threshold_label.hide()
        self.mask_interpolation_threshold_text_field.hide()

    def connect_signals(self):
        self.input_data_type_combo_box.currentTextChanged.connect(self.file_type_changed)
        self.just_save_as_nifti_check_box.stateChanged.connect(self._just_save_as_nifti_changed)
        self.use_all_structures_check_box.stateChanged.connect(self._use_all_structures_changed)
        self.run_button.clicked.connect(self.run_selection)
        self.mask_interpolation_method_combo_box.currentTextChanged.connect(
            lambda:
            (
                self.mask_interpolation_threshold_label.show(),
                self.mask_interpolation_threshold_text_field.show()
            )
            if self.mask_interpolation_method_combo_box.currentText() not in ['NN', 'Mask Interpolation:']
            else (
                self.mask_interpolation_threshold_label.hide(),
                self.mask_interpolation_threshold_text_field.hide()
            )
        )

    def _show_dicom_elements(self):
        self.dicom_structures_label.show()
        self.dicom_structures_text_field.show()
        self.dicom_structures_info_label.show()
        self.just_save_as_nifti_check_box.show()
        self.use_all_structures_check_box.show()

    def _hide_dicom_elements(self):
        self.dicom_structures_label.hide()
        self.dicom_structures_text_field.hide()
        self.dicom_structures_info_label.hide()
        self.just_save_as_nifti_check_box.hide()
        self.use_all_structures_check_box.hide()

    def _show_nifti_elements(self):
        self.nifti_structures_label.show()
        self.nifti_structures_text_field.show()
        self.nifti_structures_info_label.show()
        self.nifti_image_label.show()
        self.nifti_image_text_field.show()
        self.nifti_image_info_label.show()

    def _hide_nifti_elements(self):
        self.nifti_structures_label.hide()
        self.nifti_structures_text_field.hide()
        self.nifti_structures_info_label.hide()
        self.nifti_image_label.hide()
        self.nifti_image_text_field.hide()
        self.nifti_image_info_label.hide()

    def _hide_preprocessing_elements(self):
        self.mask_interpolation_threshold_label.hide()
        self.mask_interpolation_threshold_text_field.hide()
        self.mask_interpolation_method_combo_box.hide()
        self.resample_resolution_label.hide()
        self.resample_resolution_text_field.hide()
        self.image_interpolation_method_combo_box.hide()
        self.resample_dimension_combo_box.hide()

    def _show_preprocessing_elements(self):

        # the threshold should not be displayed when switched back, only when specific mask interpolation methods are selected:
        #self.mask_interpolation_threshold_label.show()  # TO REMOVE
        #self.mask_interpolation_threshold_text_field.show()  # TO REMOVE
        self.mask_interpolation_method_combo_box.show()
        self.resample_resolution_label.show()
        self.resample_resolution_text_field.show()
        self.image_interpolation_method_combo_box.show()
        self.resample_dimension_combo_box.show()

    def _validate_combo_selections(self):
        """Validate combo box selections."""
        required_selections = [
            ('Threads:', self.number_of_threads_combo_box),
            ('Data Type:', self.input_data_type_combo_box),
            ('Imaging Modality:', self.input_imaging_mod_combo_box)
        ]

        if not self.input_params["just_save_as_nifti"]:
            required_selections.append(('Image Interpolation:', self.image_interpolation_method_combo_box))
            required_selections.append(('Mask Interpolation:', self.mask_interpolation_method_combo_box))
            required_selections.append(('Resample Dimension:', self.resample_dimension_combo_box))

        for message, combo_box in required_selections:
            if combo_box.currentText() == message:
                warning_msg = f"Select {message.split(':')[0]}"
                raise InvalidInputParametersError(warning_msg)

    def _just_save_as_nifti_changed(self):
        if self.just_save_as_nifti_check_box.isChecked():
            self._hide_preprocessing_elements()
        else:
            self._show_preprocessing_elements()

    def _use_all_structures_changed(self):
        if self.use_all_structures_check_box.isChecked():
            self.dicom_structures_label.hide()
            self.dicom_structures_text_field.hide()
            self.dicom_structures_info_label.hide()
        else:
            self.dicom_structures_label.show()
            self.dicom_structures_text_field.show()
            self.dicom_structures_info_label.show()

    def file_type_changed(self, text):
        """Handle changes in the selected input file type."""
        if text == "DICOM":
            self._show_dicom_elements()
            self._hide_nifti_elements()
            if self.just_save_as_nifti_check_box.isChecked():
                self._hide_preprocessing_elements()
            if self.use_all_structures_check_box.isChecked():
                self.dicom_structures_label.hide()
                self.dicom_structures_text_field.hide()
                self.dicom_structures_info_label.hide()
        elif text == "NIfTI":
            self._show_nifti_elements()
            self._hide_dicom_elements()
            self._show_preprocessing_elements()
        else:
            self._hide_dicom_elements()
            self._hide_nifti_elements()

    def check_input_parameters(self):
        # Validate combo box selections
        self._validate_combo_selections()
        self.check_common_input_parameters()

        self.input_params["nifti_image_name"] = self.get_text_from_text_field(self.input_params["nifti_image_name"])
        self.input_params["nifti_structures"] = self.get_list_from_text_field(self.input_params["nifti_structures"])
        self.input_params["dicom_structures"] = self.get_list_from_text_field(self.input_params["dicom_structures"])

        if self.input_params["just_save_as_nifti"]:
            self.input_params["resample_dimension"] = None
            self.input_params["resample_resolution"] = None
            self.input_params["image_interpolation_method"] = None
            self.input_params["mask_interpolation_method"] = None
            self.input_params["mask_interpolation_threshold"] = None
        else:
            try:
                self.input_params["resample_resolution"] = float(self.input_params["resample_resolution"])
            except ValueError:
                msg = "Select valid resample resolution"
                raise InvalidInputParametersError(msg)

            if self.input_params["mask_interpolation_method"] != "NN":
                try:
                    self.input_params["mask_interpolation_threshold"] = float(self.input_params["mask_interpolation_threshold"])
                except ValueError:
                    msg = "Select valid mask interpolation threshold"
                    raise InvalidInputParametersError(msg)
            else:
                self.input_params["mask_interpolation_threshold"] = 0.5

    def get_input_parameters(self):
        """Collect input parameters from UI elements."""
        input_parameters = {
            'input_directory': self.load_dir_text_field.text(),
            'start_folder': self.start_folder_text_field.text(),
            'stop_folder': self.stop_folder_text_field.text(),
            'list_of_patient_folders': self.list_of_patient_folders_text_field.text(),
            'input_data_type': self.input_data_type_combo_box.currentText(),
            'output_directory': self.save_dir_text_field.text(),
            'number_of_threads': self.number_of_threads_combo_box.currentText(),
            'dicom_structures': self.dicom_structures_text_field.text(),
            'nifti_image_name': self.nifti_image_text_field.text(),
            'nifti_structures': self.nifti_structures_text_field.text(),
            'resample_resolution': self.resample_resolution_text_field.text(),
            'image_interpolation_method': self.image_interpolation_method_combo_box.currentText(),
            'resample_dimension': self.resample_dimension_combo_box.currentText(),
            'mask_interpolation_method': self.mask_interpolation_method_combo_box.currentText(),
            'mask_interpolation_threshold': self.mask_interpolation_threshold_text_field.text(),
            'input_imaging_modality': self.input_imaging_mod_combo_box.currentText(),
            'just_save_as_nifti': self.just_save_as_nifti_check_box.checkState(),
            'use_all_structures': self.use_all_structures_check_box.checkState(),
            'mask_union': self.mask_union_check_box.checkState(),
        }
        self.input_params = input_parameters

    def run_selection(self):
        """Executes preprocessing based on user-selected options."""
        close_all_loggers()
        self.logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.logger = get_logger(self.logger_date_time + '_Preprocessing')
        self.logger.info("Preprocessing started")

        # Prepare input parameters for radiomics extraction
        self.get_input_parameters()

        # Check input parameters
        try:
            self.check_input_parameters()
        except InvalidInputParametersError as e:
            # Stop execution if input parameters are invalid
            self.logger.error(e)
            CustomWarningBox(str(e)).response()
            return

        # Get patient folders
        list_of_patient_folders = self.get_patient_folders()

        # Determine structure set based on data type
        structure_set = None
        if self.input_params["input_data_type"].lower() == "nifti":
            structure_set = self.input_params.get("nifti_structures")
        elif self.input_params["input_data_type"].lower() == "dicom":
            if not self.input_params["use_all_structures"]:
                structure_set = self.input_params["dicom_structures"]

        # Process each patient folder
        if list_of_patient_folders:
            Parallel(n_jobs=self.input_params["number_of_threads"])(
                delayed(process_patient_folder)(self.input_params, patient_folder, structure_set) for patient_folder in list_of_patient_folders)
        else:
            CustomWarningBox("No patients to calculate preprocess from.")

        self.logger.info("Preprocessing finished!")
        CustomInfoBox("Preprocessing finished!").response()

    def save_settings(self):
        """
        Update specific fields in the config.json file without overwriting existing data.
        """
        # Data to be updated
        self.get_input_parameters()
        data = {'prep_' + key: value for key, value in self.input_params.items()}
        file_path = os.path.join(os.getcwd(), 'config.json')

        # Attempt to read the existing data from the file
        try:
            with open(file_path, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = {}

        # Update the existing data with the new data
        existing_data.update(data)

        # Write the updated data back to the file
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)

    def load_settings(self):
        """
        Load input data from a JSON file.
        """
        file_path = os.path.join(os.getcwd(), 'config.json')
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.load_dir_text_field.setText(data.get('prep_input_directory', ''))
                self.start_folder_text_field.setText(data.get('prep_start_folder', ''))
                self.stop_folder_text_field.setText(data.get('prep_stop_folder', ''))
                self.list_of_patient_folders_text_field.setText(data.get('prep_list_of_patient_folders', ''))
                self.input_data_type_combo_box.setCurrentText(data.get('prep_input_data_type', 'Data Type:'))
                self.save_dir_text_field.setText(data.get('prep_output_directory', ''))
                self.number_of_threads_combo_box.setCurrentText(data.get('prep_number_of_threads', 'Threads:'))
                self.dicom_structures_text_field.setText(data.get('prep_dicom_structures', ''))
                self.nifti_image_text_field.setText(data.get('prep_nifti_image_name', ''))
                self.nifti_structures_text_field.setText(data.get('prep_nifti_structures', ''))
                self.resample_resolution_text_field.setText(data.get('prep_resample_resolution', ''))
                self.image_interpolation_method_combo_box.setCurrentText(
                    data.get('prep_image_interpolation_method', 'Image Interpolation:'))
                self.resample_dimension_combo_box.setCurrentText(data.get('prep_resample_dimension', 'Resample Dimension:'))
                self.mask_interpolation_method_combo_box.setCurrentText(
                    data.get('prep_mask_interpolation_method', 'Mask Interpolation:'))
                self.mask_interpolation_threshold_text_field.setText(
                    data.get('prep_mask_interpolation_threshold', '0.5'))
                self.input_imaging_mod_combo_box.setCurrentText(
                    data.get('prep_input_imaging_modality', 'Imaging Modality:'))
                self.just_save_as_nifti_check_box.setCheckState(data.get('prep_just_save_as_nifti', 0))
                self.use_all_structures_check_box.setCheckState(data.get('prep_use_all_structures', 0))
                self.mask_union_check_box.setCheckState(data.get('prep_mask_union', 0))

        except FileNotFoundError:
            print("No previous data found!")
