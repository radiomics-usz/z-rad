import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ._base_tab import BaseTab, load_images, load_mask
from .toolbox_gui import CustomLabel, CustomBox, CustomTextField, CustomCheckBox, CustomWarningBox, CustomInfo, CustomInfoBox
from ..logic.exceptions import InvalidInputParametersError, DataStructureError
from ..logic.image import get_dicom_files, get_all_structure_names
from ..logic.radiomics import Radiomics
from ..logic.toolbox_logic import get_logger, close_all_loggers

logging.captureWarnings(True)


def process_patient_folder(input_params, patient_folder, structure_set):
    # Logger
    logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logger = get_logger(logger_date_time + '_Radiomics')

    def is_slice_weighting():
        """Determines if slice weighting is applied based on current settings."""
        aggr_dim = input_params["agr_strategy"].split(',')[0]
        weighting = input_params["weighting"]
        if aggr_dim == '2D':
            return weighting == 'Weighted Mean'
        return False

    def is_slice_median():
        """Determines if slice median is applied based on current settings."""
        aggr_dim = input_params["agr_strategy"].split(',')[0]
        weighting = input_params["weighting"]
        if aggr_dim == '2D':
            return weighting == 'Median'
        return False

    # Initialize Radiomics instance
    rad_instance = Radiomics(
        aggr_dim=input_params['aggregation_method'][0],
        aggr_method=input_params['aggregation_method'][1],
        intensity_range=input_params['intensity_range'],
        outlier_range=input_params['outlier_range'],
        number_of_bins=input_params['discretization'][1],
        bin_size=input_params['discretization'][2],
        slice_weighting=is_slice_weighting(),
        slice_median=is_slice_median(),
    )

    logger.info(f"Processing patient: {patient_folder}.")
    try:
        if input_params["nifti_filtered_image_name"]:
            image, filtered_image = load_images(input_params, patient_folder)
        else:
            image = load_images(input_params, patient_folder)
            filtered_image = None
    except DataStructureError as e:
        logger.error(e)

    if input_params["use_all_structures"]:
        input_directory = os.path.join(input_params["input_directory"], patient_folder)
        rtstruct_path = get_dicom_files(input_directory, modality='RTSTRUCT')[0]
        structure_set = get_all_structure_names(rtstruct_path)

    radiomic_features_list = []
    for mask_name in structure_set:
        mask = load_mask(input_params, patient_folder, mask_name, image)
        if mask and mask.array is not None:
            logger.info(f"Processing patient: {patient_folder} with ROI: {mask_name}.")
            try:
                rad_instance.extract_features(image, mask, filtered_image)
            except DataStructureError as e:
                logger.error(e)
                logger.info(f"Patient {patient_folder} with mask {mask_name} skipped.")
                continue
            radiomic_features = rad_instance.features_
            radiomic_features['pat_id'] = patient_folder
            radiomic_features['mask_id'] = mask_name
            radiomic_features_list.append(radiomic_features)

    return radiomic_features_list


class RadiomicsTab(BaseTab):
    """Tab for configuring radiomics extraction options."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_dicom_elements()
        self.init_nifti_elements()
        self.init_radiomics_elements()
        self.connect_signals()

    def init_dicom_elements(self):
        """Initialize UI components related to DICOM elements."""
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

        # Filtered image
        self.nifti_filtered_image_label = CustomLabel(
            'NIfTI Filtered Image:',
            900, 300, 200, 50, self,
            style="color: white;"
        )
        self.nifti_filtered_image_text_field = CustomTextField(
            "E.g. filtered_imageCT",
            1060, 300, 120, 50, self
        )
        self.nifti_filtered_image_info_label = CustomInfo(
            ' i',
            'Specify filtered NIfTI image file without file extension'
            '\nIf radiomics are extracted from a filtered NIfTI image, you must provide both the filtered and the original NIfTI images.'
            '\nPlease specify the file names of the NIfTI images without their extensions.',
            1190, 300, 14, 14, self
        )

        # Hide NIfTI elements
        self._hide_nifti_elements()

    def init_radiomics_elements(self):
        """Initialize UI components related to radiomics processing options."""
        # Outlier detection
        self.outlier_detection_check_box = CustomCheckBox(
            'Outlier Removal (in \u03C3)',
            200, 460, 250, 50, self
        )
        self.outlier_detection_text_field = CustomTextField(
            "E.g. 3",
            410, 460, 100, 50, self
        )
        self.outlier_detection_text_field.hide()

        # Intensity range
        self.intensity_range_text_field = CustomTextField(
            "E.g. -1000, 400",
            410, 375, 210, 50, self
        )
        self.intensity_range_text_field.hide()
        self.intensity_range_check_box = CustomCheckBox(
            'Intensity Range',
            200, 380, 200, 50, self)

        # Discretization
        self.discretization_combo_box = CustomBox(
            700, 460, 170, 50, self,
            item_list=[
                "Discretization:", "Number of Bins", "Bin Size"
            ]
        )
        self.bin_number_text_field = CustomTextField(
            "E.g. 5",
            1000, 460, 100, 50, self
        )
        self.bin_size_text_field = CustomTextField("E.g. 50", 1000, 460, 100, 50, self)
        self.bin_number_text_field.hide()
        self.bin_size_text_field.hide()

        # Aggregation
        self.aggr_dim_and_method_combo_box = CustomBox(
            700, 375, 250, 50, self,
            item_list=[
                "Texture Aggregation Method:",
                "2D, averaged",
                "2D, slice-merged",
                "2.5D, direction-merged",
                "2.5D, merged",
                "3D, averaged",
                "3D, merged"
            ]
        )
        self.weighting_combo_box = CustomBox(
            1000, 375, 175, 50, self,
            item_list=[
                "Slice Averaging:", "Mean", "Weighted Mean", "Median"]
        )
        self.weighting_combo_box_info_label = CustomInfo(
            ' i',
            "'Mean' approach is in agreement with IBSI. \n'Weighted Mean' and 'Median' "
            "are custom solutions developed at USZ.",
            1185, 375, 14, 14, self
        )
        self.weighting_combo_box_info_label.hide()
        self.weighting_combo_box.hide()

    def connect_signals(self):
        """Connect signals for UI elements."""
        self.input_data_type_combo_box.currentTextChanged.connect(self.file_type_changed)
        self.use_all_structures_check_box.stateChanged.connect(self._use_all_structures_changed)
        self.outlier_detection_check_box.stateChanged.connect(
            lambda: self.outlier_detection_text_field.setVisible(self.outlier_detection_check_box.isChecked())
        )
        self.intensity_range_check_box.stateChanged.connect(
            lambda: self.intensity_range_text_field.setVisible(self.intensity_range_check_box.isChecked())
        )
        self.discretization_combo_box.currentTextChanged.connect(self._toggle_discretization_visibility)
        self.aggr_dim_and_method_combo_box.currentTextChanged.connect(self._toggle_weighting_visibility)
        self.run_button.clicked.connect(self.run_selection)

    def file_type_changed(self, text):
        """Handle changes in the selected input file type."""
        if text == "DICOM":
            self._show_dicom_elements()
            self._hide_nifti_elements()
            if self.use_all_structures_check_box.isChecked():
                self.dicom_structures_label.hide()
                self.dicom_structures_text_field.hide()
                self.dicom_structures_info_label.hide()
        elif text == "NIfTI":
            self._show_nifti_elements()
            self._hide_dicom_elements()
        else:
            self._hide_dicom_elements()
            self._hide_nifti_elements()

    def check_input_parameters(self):
        # Validate combo box selections
        self._validate_combo_selections()
        self.check_common_input_parameters()

        self.input_params["nifti_image_name"] = self.get_text_from_text_field(self.input_params["nifti_image_name"])
        self.input_params["nifti_filtered_image_name"] = self.get_text_from_text_field(self.input_params["nifti_filtered_image_name"])
        self.input_params["nifti_structures"] = self.get_list_from_text_field(self.input_params["nifti_structures"])
        self.input_params["dicom_structures"] = self.get_list_from_text_field(self.input_params["dicom_structures"])

        self.input_params["outlier_range"] = self._get_outlier_sigma()
        self.input_params["intensity_range"] = self._get_intensity_range()
        self.input_params["discretization"] = self._get_discretization_settings()
        self.input_params["aggregation_method"] = self._get_aggregation_settings()

    def get_input_parameters(self):
        """Collect input parameters from UI elements."""
        input_parameters = {
            'input_directory': self.load_dir_text_field.text(),
            'start_folder': self.start_folder_text_field.text(),
            'stop_folder': self.stop_folder_text_field.text(),
            'list_of_patient_folders': self.list_of_patient_folders_text_field.text(),
            'input_data_type': self.input_data_type_combo_box.currentText(),
            'input_imaging_modality': self.input_imaging_mod_combo_box.currentText(),
            'output_directory': self.save_dir_text_field.text(),
            'number_of_threads': self.number_of_threads_combo_box.currentText(),
            'dicom_structures': self.dicom_structures_text_field.text(),
            'nifti_structures': self.nifti_structures_text_field.text(),
            'nifti_image_name': self.nifti_image_text_field.text(),
            'nifti_filtered_image_name': self.nifti_filtered_image_text_field.text(),
            'agr_strategy': self.aggr_dim_and_method_combo_box.currentText(),
            'binning': self.discretization_combo_box.currentText(),
            'number_of_bins': self.bin_number_text_field.text(),
            'bin_size': self.bin_size_text_field.text(),
            'intensity_range_check_box': self.intensity_range_check_box.checkState(),
            'intensity_range': self.intensity_range_text_field.text(),
            'outlier_detection_check_box': self.outlier_detection_check_box.checkState(),
            'outlier_detection_value': self.outlier_detection_text_field.text(),
            'weighting': self.weighting_combo_box.currentText(),
            'use_all_structures': self.use_all_structures_check_box.checkState(),
        }
        self.input_params = input_parameters

    def run_selection(self):
        """Executes radiomics extraction based on user-selected options."""
        close_all_loggers()
        self.logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.logger = get_logger(self.logger_date_time + '_Radiomics')
        self.logger.info("Radiomics started")

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
        if self.input_params["input_data_type"] == "nifti":
            structure_set = self.input_params.get("nifti_structures")
        elif self.input_params["input_data_type"] == "dicom":
            if not self.input_params["use_all_structures"]:
                structure_set = self.input_params["dicom_structures"]

        # Process each patient folder
        if list_of_patient_folders:
            radiomic_features_list = Parallel(n_jobs=self.input_params["number_of_threads"])(
                delayed(process_patient_folder)(self.input_params, patient_folder, structure_set) for patient_folder in list_of_patient_folders)

            # Save features to CSV
            if radiomic_features_list:
                radiomic_features_list = [item for sublist in radiomic_features_list for item in sublist]
                radiomic_features_df = pd.DataFrame(radiomic_features_list)
                radiomic_features_df.set_index(['pat_id', 'mask_id'], inplace=True)
                file_path = os.path.join(self.input_params["output_directory"], 'radiomics.csv')
                radiomic_features_df.to_csv(file_path)
                self.logger.info(f"Radiomics saved to {file_path}.")
        else:
            CustomWarningBox("No patients to calculate radiomics from.")

        self.logger.info("Radiomics finished!")
        CustomInfoBox("Radiomics finished!").response()

    def _get_outlier_sigma(self):
        """
        Retrieve the outlier detection range from the text field if the checkbox is checked.

        Returns:
            float | None: A float representing the outlier range or None if the checkbox is unchecked.
        """
        # Return None if the outlier detection checkbox is not checked
        if self.outlier_detection_check_box.isChecked():
            # Get the text from the outlier detection text field and strip any whitespace
            text = self.outlier_detection_text_field.text().strip()

            # Check if the text field is empty or contains only whitespace
            if not text:
                warning_msg = "Enter standard deviation for outlier filtering."
                raise InvalidInputParametersError(warning_msg)

            # Convert the text field input to a float, handling any conversion errors
            try:
                outlier_sigma = float(text)
            except ValueError:
                # Handle any non-numeric input gracefully
                warning_msg = "Invalid input for the standard deviation in outlier filtering. Please enter a valid number."
                raise InvalidInputParametersError(warning_msg)

            if outlier_sigma < 0:
                warning_msg = "The standard deviation in outlier filtering needs to be a positive number."
                raise InvalidInputParametersError(warning_msg)

            return outlier_sigma
        else:
            return None

    def _get_intensity_range(self):
        """
        Retrieve the intensity range from the text field if the checkbox is checked.

        Returns:
            list[float] | None: A list of floats representing the intensity range or None if the checkbox is unchecked.
        """
        # Return None if the intensity range checkbox is not checked
        if not self.intensity_range_check_box.isChecked():
            return None

        # Get the text from the intensity range text field and strip any whitespace
        text = self.intensity_range_text_field.text().strip()

        # Check if the text field is empty or contains only whitespace
        if not text:
            warning_msg = "Enter intensity range"
            raise InvalidInputParametersError(warning_msg)

        # Split the text field input by commas, convert to floats, handling empty values as np.inf
        try:
            intensity_range = [
                np.inf if value.strip() == '' else float(value.strip())
                for value in text.split(',')
            ]
        except ValueError:
            # Handle any non-numeric input gracefully
            warning_msg = "Invalid input for intensity range. Please enter numbers separated by a comma."
            raise InvalidInputParametersError(warning_msg)

        if len(intensity_range) != 2:
            warning_msg = "Intensity range needs to be a list containing two numbers separated by a comma."
            raise InvalidInputParametersError(warning_msg)
        return intensity_range

    def _get_discretization_settings(self):
        """
        Retrieve discretization settings based on user input.

        Returns:
            tuple: A tuple containing (discretization_method, number_of_bins, bin_size).
                   Returns (None, None, None) if required input is missing or invalid.
        """
        def get_bin_input(text_field, warning_message, expected_type):
            """
            Helper function to get and validate the bin input from a text field.

            Args:
                text_field (QLineEdit): The text field from which to retrieve the input.
                warning_message (str): The warning message to display if input is missing or invalid.
                expected_type (type): The expected type of the input (int or float).

            Returns:
                int | float | None: The input converted to the expected type, or None if invalid.
            """
            # Get text input and strip whitespace
            text = text_field.text().strip()

            # Check if input is empty
            if not text:
                raise InvalidInputParametersError(warning_message)

            # Attempt to convert the input to the expected type
            try:
                return expected_type(text)
            except ValueError:
                warning_msg = f"Invalid input. Please enter a valid {expected_type.__name__}."
                raise InvalidInputParametersError(warning_msg)

        # Retrieve the selected discretization method from the combo box
        discretization_method = self.discretization_combo_box.currentText()

        # Initialize bins-related variables
        number_of_bins = None
        bin_size = None

        # Check the selected discretization method and retrieve the relevant input
        if discretization_method == 'Number of Bins':
            # Get and validate the number of bins input
            number_of_bins = get_bin_input(self.bin_number_text_field, "Enter Number of Bins", int)
            if number_of_bins is None:
                warning_msg = f"Enter Number of Bins."
                raise InvalidInputParametersError(warning_msg)

        elif discretization_method == 'Bin Size':
            # Get and validate the bin size input
            bin_size = get_bin_input(self.bin_size_text_field, "Enter Bin Size", float)
            if bin_size is None:
                warning_msg = f"Enter Bin Size"
                raise InvalidInputParametersError(warning_msg)

        return discretization_method, number_of_bins, bin_size

    def _get_aggregation_settings(self):
        """
        Retrieve aggregation settings from the combo box.

        Returns:
            tuple: A tuple containing (aggregation_dimension, aggregation_method).
                   Returns (None, None) if the input is missing or invalid.
        """

        def map_aggregation_method(method):
            """Maps aggregation method to its corresponding abbreviation."""
            method_mapping = {
                'merged': 'MERG',
                'averaged': 'AVER',
                'slice-merged': 'SLICE_MERG',
                'direction-merged': 'DIR_MERG'
            }
            return method_mapping.get(method, method)

        # Retrieve the current text from the combo box
        combo_text = self.aggr_dim_and_method_combo_box.currentText()

        # Split the text to get dimension and method
        try:
            aggr_dim, aggr_method = map(str.strip, combo_text.split(','))
        except ValueError:
            warning_msg = f"Invalid aggregation settings format. Please ensure it's in 'dimension,method' format."
            raise InvalidInputParametersError(warning_msg)

        # Map the aggregation method to its corresponding value
        aggr_method = map_aggregation_method(aggr_method)

        return aggr_dim, aggr_method

    def _validate_combo_selections(self):
        """Validate combo box selections."""
        required_selections = [
            ('Threads:', self.number_of_threads_combo_box),
            ('Data Type:', self.input_data_type_combo_box),
            ('Discretization:', self.discretization_combo_box),
            ('Texture Features Aggr. Method:', self.aggr_dim_and_method_combo_box),
            ('Imaging Modality:', self.input_imaging_mod_combo_box)
        ]
        for message, combo_box in required_selections:
            if combo_box.currentText() == message:
                warning_msg = f"Select {message.split(':')[0]}"
                raise InvalidInputParametersError(warning_msg)

    def _show_dicom_elements(self):
        self.dicom_structures_label.show()
        self.dicom_structures_text_field.show()
        self.dicom_structures_info_label.show()
        self.use_all_structures_check_box.show()

    def _hide_dicom_elements(self):
        self.dicom_structures_label.hide()
        self.dicom_structures_text_field.hide()
        self.dicom_structures_info_label.hide()
        self.use_all_structures_check_box.hide()

    def _show_nifti_elements(self):
        self.nifti_structures_label.show()
        self.nifti_structures_text_field.show()
        self.nifti_structures_info_label.show()
        self.nifti_image_label.show()
        self.nifti_image_text_field.show()
        self.nifti_image_info_label.show()
        self.nifti_filtered_image_label.show()
        self.nifti_filtered_image_text_field.show()
        self.nifti_filtered_image_info_label.show()

    def _hide_nifti_elements(self):
        self.nifti_structures_label.hide()
        self.nifti_structures_text_field.hide()
        self.nifti_structures_info_label.hide()
        self.nifti_image_label.hide()
        self.nifti_image_text_field.hide()
        self.nifti_image_info_label.hide()
        self.nifti_filtered_image_label.hide()
        self.nifti_filtered_image_text_field.hide()
        self.nifti_filtered_image_info_label.hide()

    def _toggle_discretization_visibility(self, text):
        """Update UI based on selected discretization method."""
        if text == "Number of Bins":
            self.bin_number_text_field.show()
            self.bin_size_text_field.hide()
        elif text == "Bin Size":
            self.bin_size_text_field.show()
            self.bin_number_text_field.hide()
        else:
            self.bin_number_text_field.hide()
            self.bin_size_text_field.hide()

    def _toggle_weighting_visibility(self, text):
        """Update UI based on selected aggregation dimension."""
        if text in ["2D, averaged", "2D, slice-merged"]:
            self.weighting_combo_box.show()
            self.weighting_combo_box_info_label.show()
        else:
            self.weighting_combo_box.hide()
            self.weighting_combo_box_info_label.hide()

    def save_settings(self):
        """
        Update specific radiology-related fields in the config.json file without
        overwriting existing data.
        """
        self.get_input_parameters()
        data = {'radiomics_' + key: value for key, value in self.input_params.items()}
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
        file_path = os.path.join(os.getcwd(), 'config.json')
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.load_dir_text_field.setText(data.get('radiomics_input_directory', ''))
                self.start_folder_text_field.setText(data.get('radiomics_start_folder', ''))
                self.stop_folder_text_field.setText(data.get('radiomics_stop_folder', ''))
                self.list_of_patient_folders_text_field.setText(data.get('radiomics_list_of_patient_folders', ''))
                self.input_data_type_combo_box.setCurrentText(data.get('radiomics_input_data_type', 'Data Type:'))
                self.save_dir_text_field.setText(data.get('radiomics_output_directory', ''))
                self.number_of_threads_combo_box.setCurrentText(data.get('radiomics_number_of_threads', 'Threads:'))
                self.nifti_structures_text_field.setText(data.get('radiomics_nifti_structures', ''))
                self.dicom_structures_text_field.setText(data.get('radiomics_dicom_structures', ''))
                self.nifti_image_text_field.setText(data.get('radiomics_nifti_image_name', ''))
                self.nifti_filtered_image_text_field.setText(data.get('radiomics_nifti_filtered_image_name', ''))
                self.intensity_range_text_field.setText(data.get('radiomics_intensity_range', ''))
                self.aggr_dim_and_method_combo_box.setCurrentText(
                    data.get('radiomics_agr_strategy', 'Texture Features Aggr. Method:'))
                self.discretization_combo_box.setCurrentText(data.get('radiomics_binning', 'Discretization:'))
                self.bin_number_text_field.setText(data.get('radiomics_number_of_bins', ''))
                self.bin_size_text_field.setText(data.get('radiomics_bin_size', ''))
                self.intensity_range_check_box.setCheckState(data.get('radiomics_intensity_range_check_box', 0))
                self.outlier_detection_check_box.setCheckState(data.get('radiomics_outlier_detection_check_box', 0))
                self.outlier_detection_text_field.setText(data.get('radiomics_outlier_detection_value', ''))
                self.weighting_combo_box.setCurrentText(data.get('radiomics_weighting', 'Slice Averaging:'))
                self.input_imaging_mod_combo_box.setCurrentText(data.get('radiomics_input_imaging_modality', 'Imaging Modality:'))
                self.use_all_structures_check_box.setCheckState(data.get('radiomics_use_all_structures', 0))
        except FileNotFoundError:
            print("No previous data found!")

    def _use_all_structures_changed(self):
        if self.use_all_structures_check_box.isChecked():
            self.dicom_structures_label.hide()
            self.dicom_structures_text_field.hide()
            self.dicom_structures_info_label.hide()
        else:
            self.dicom_structures_label.show()
            self.dicom_structures_text_field.show()
            self.dicom_structures_info_label.show()