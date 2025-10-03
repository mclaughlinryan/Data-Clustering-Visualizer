"""
Data clustering visualizer
A graphical user interface application that loads a selected data file and performs and visualizes data clustering on the data.
The user can cluster the data by selecting from several different clustering algorithms and choosing the number of clusters to cluster the data into for certain algorithms.
The clustering is done for numeric data and the program has different options for handling non-numeric values to convert the data into numeric form.
The data can be saved as text in a text or CSV file as well as saving a graph of the clustering.
"""

import os
import sys
import numpy as np
from PyQt6.QtGui import QIntValidator, QDoubleValidator
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget, QLabel, QMainWindow, QPushButton, QFileDialog, QComboBox, QLineEdit, QRadioButton, QSizePolicy, QHBoxLayout, QFrame, QCheckBox, QButtonGroup
from sklearn.cluster import KMeans, MeanShift, DBSCAN, HDBSCAN, AgglomerativeClustering, AffinityPropagation, SpectralClustering, Birch, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class DataClusteringVisualizerInterface(QMainWindow):
    """Graphical user interface for performing and visualizing data clustering on selected data."""
    def __init__(self):
        """Constructs the graphical user interface and initializes flag variables that determine whether or not
        the conditions are met for data clustering to be performed and graphed."""

        # Initialization of graphical user interface window
        super(DataClusteringVisualizerInterface, self).__init__()

        # Set size, title, and layout of graphical user interface window
        self.resize(600, 450)
        self.setWindowTitle("Data Clustering Visualizer")
        self.window_layout = QHBoxLayout()
        self.window_widget = QWidget()
        self.window_widget.setLayout(self.window_layout)

        # Side panel component with data clustering options
        self.side_panel_layout = QVBoxLayout()
        self.side_panel_widget = QWidget()
        self.side_panel_widget.setLayout(self.side_panel_layout)
        self.side_panel_widget.setFixedWidth(int((47/120)*self.width()))

        # Button for selecting data file with data to be clustered
        self.load_data_button = QPushButton("Load Data")
        self.load_data_button.setAccessibleName("Load Data")
        self.load_data_button.clicked.connect(self.load_data)
        self.load_data_button.setFixedWidth(90)

        # Option for whether or not the data file contains class assignments for each data point
        self.data_with_classes_check_box = QCheckBox("Data contains classes")
        self.data_with_classes_check_box.setAccessibleName("Data Classes Option")
        self.data_with_classes_check_box.stateChanged.connect(self.data_with_classes_option_changed)

        # Error message if the data has any errors
        self.file_data_error_label = QLabel()
        self.file_data_error_label.setAccessibleName("File Data Error")

        # Displays the number of classes in the data if the data has class assignments
        self.data_classes_label = QLabel()
        self.data_classes_label.setAccessibleName("Data Classes Count")

        # Displays text if the data contains non-numeric values
        self.data_non_numbers_label = QLabel("The data has non-numeric values.\nSelect an option below for how to\nhandle non-numeric values.")
        self.data_non_numbers_label.setAccessibleName("Data Non Numbers Text")

        # Options for converting non-numeric values in the data to numeric values
        self.data_non_numbers_option_combo_box = QComboBox()
        self.data_non_numbers_option_combo_box.addItems(["", "Assign all non-numeric values to 0", "Assign each non-numeric value to a number", "Exclude data points with non-numeric values", "Exclude features with non-numeric values"])
        self.data_non_numbers_option_combo_box.setAccessibleName("Data Non Numbers Options")
        self.data_non_numbers_option_combo_box.currentIndexChanged.connect(self.data_non_numbers_option_changed)
        self.data_non_numbers_layout = QHBoxLayout()
        self.data_non_numbers_widget = QWidget()
        self.data_non_numbers_widget.setLayout(self.data_non_numbers_layout)
        self.data_non_numbers_widget.setAccessibleName("Data Non Numbers Values Container")
        self.data_non_numbers_layout.setContentsMargins(0, 0, 0, 0)
        self.data_non_numbers_layout.setSpacing(0)
        self.data_non_numbers_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.feature_values_non_numbers_invalid_input = ["-", "+", ".", "-.", "+."]

        # Clustering algorithms that can be selected for clustering the data
        self.clustering_algorithm_select_label = QLabel("Clustering algorithm:")
        self.clustering_algorithm_combo_box = QComboBox()
        self.clustering_algorithm_combo_box.addItems(["", "K-Means", "Mean Shift", "DBSCAN", "HDBSCAN", "Gaussian Mixture Models", "Agglomerative", "Affinity Propagation", "Spectral", "BIRCH", "OPTICS"])
        self.clustering_algorithm_combo_box.setAccessibleName("Clustering Algorithm Options")
        self.clustering_algorithm_combo_box.currentIndexChanged.connect(self.clustering_algorithm_changed)
        self.clustering_algorithms_no_num_clusters_input = ["Mean Shift", "DBSCAN", "HDBSCAN", "Affinity Propagation", "Spectral", "BIRCH", "OPTICS"]

        # Option for entering in the number of clusters to cluster the data into for certain algorithms
        self.num_clusters_label = QLabel("Number of clusters:")
        self.num_clusters_label.setAccessibleName("Number of Clusters Text")
        self.num_clusters_input = QLineEdit()
        intInput = QIntValidator()
        intInput.setRange(0, 999)
        self.num_clusters_input.setValidator(intInput)
        self.num_clusters_input.setAccessibleName("Number of Clusters Input")
        self.num_clusters_input.textChanged.connect(self.num_clusters_changed)
        self.num_clusters_input.setFixedWidth(80)

        # Option to display the clustered data as a 2D or 3D graph
        self.display_dimensions_label = QLabel("Display dimensionality:")
        self.display_dimensions_layout = QHBoxLayout()
        self.display_dimensions_widget = QWidget()
        self.display_dimensions_widget.setLayout(self.display_dimensions_layout)
        self.display_dimensions_button_group = QButtonGroup()
        self.button2D = QRadioButton("2D")
        self.button2D.text = "2D"
        self.button2D.toggled.connect(self.radio_button_selected)
        self.button3D = QRadioButton("3D")
        self.button3D.text = "3D"
        self.button3D.toggled.connect(self.radio_button_selected)
        self.display_dimensions_button_group.addButton(self.button2D)
        self.display_dimensions_button_group.addButton(self.button3D)
        self.display_dimensions_layout.addWidget(self.button2D)
        self.display_dimensions_layout.addWidget(self.button3D)
        self.display_dimensions_layout.setContentsMargins(0, 0, 0, 0)
        self.display_dimensions_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Buttons for displaying the graph of the clustered data in one of two windows to allow
        # for two graphs to be displayed at the same time and for comparison of data clustering
        self.display_data_clustering_label = QLabel("Display data clustering")
        self.display_data_clustering_window_layout = QHBoxLayout()
        self.display_data_clustering_window_widget = QWidget()
        self.display_data_clustering_window_widget.setLayout(self.display_data_clustering_window_layout)
        self.display_data_clustering_window1_button = QPushButton("Display 1")
        self.display_data_clustering_window1_button.setAccessibleName("Show Display 1")
        self.display_data_clustering_window1_button.clicked.connect(self.display_data_clustering)
        self.display_data_clustering_window2_button = QPushButton("Display 2")
        self.display_data_clustering_window2_button.setAccessibleName("Show Display 2")
        self.display_data_clustering_window2_button.clicked.connect(self.display_data_clustering)
        self.display_data_clustering_window_layout.addWidget(self.display_data_clustering_window1_button)
        self.display_data_clustering_window_layout.addWidget(self.display_data_clustering_window2_button)
        self.display_data_clustering_window_layout.setContentsMargins(0, 0, 0, 0)
        self.display_data_clustering_window_layout.setSpacing(8)
        self.display_data_clustering_window_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.side_panel_layout.addWidget(self.load_data_button)
        self.side_panel_layout.addWidget(self.data_with_classes_check_box)
        self.side_panel_layout.addWidget(self.clustering_algorithm_select_label)
        self.side_panel_layout.addWidget(self.clustering_algorithm_combo_box)
        self.side_panel_layout.addWidget(self.display_dimensions_label)
        self.side_panel_layout.addWidget(self.display_dimensions_widget)
        self.side_panel_layout.addWidget(self.display_data_clustering_label)
        self.side_panel_layout.addWidget(self.display_data_clustering_window_widget)
        self.side_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.side_panel_layout.setSpacing(8)
        self.side_panel_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.display_data_clustering_window1_button.setEnabled(False)
        self.display_data_clustering_window2_button.setEnabled(False)

        self.removable_widgets_load_data = [self.file_data_error_label.accessibleName(), self.data_classes_label.accessibleName(), self.data_non_numbers_label.accessibleName(), self.data_non_numbers_option_combo_box.accessibleName(), self.data_non_numbers_widget.accessibleName(), self.num_clusters_label.accessibleName(), self.num_clusters_input.accessibleName()]
        self.removable_widgets_process_data = [self.data_classes_label.accessibleName(), self.data_non_numbers_label.accessibleName(), self.data_non_numbers_option_combo_box.accessibleName(), self.data_non_numbers_widget.accessibleName()]

        # Initializing flag variables for conditions to be met for data clustering to the false state
        self.display_dimension = ""
        self.data_imported = False
        self.file_data_error = False
        self.data_with_classes = False
        self.data_non_numbers = False
        self.feature_values_non_numbers_entered = False
        self.feature_values_non_numbers_entered_not_applicable = False
        self.clustering_algorithm_selected = False
        self.num_clusters_entered = False
        self.num_clusters_not_applicable = False
        self.dimension_selected = False

        # Line to separate panel with data clustering options from display of data clustering graphs
        self.separator_line = QFrame()
        self.separator_line.setAccessibleName("Separator Line")
        self.separator_line.setFrameShape(QFrame.Shape.VLine)
        self.separator_line.setLineWidth(1)

        self.window_layout.addWidget(self.side_panel_widget)
        self.window_layout.addWidget(self.separator_line)
        self.window_layout.setContentsMargins(7, 7, 7, 7)
        self.window_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.display_window1 = False
        self.display_window2 = False

        self.setCentralWidget(self.window_widget)

        self.window_dimensions = [self.width(), self.height()]

    def load_data(self):
        """Loads a selected data file containing data points with features and possible class assignments,
        resets the graphical user interface and flag variables, checks for errors with the data,
        and calls the function for processing the data if the data does not have any errors."""
        dialog = QFileDialog(self)
        dialog.setDirectory(os.path.dirname(os.path.realpath(__file__)))
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Plain Text (*.txt);; Comma Separated Values (*.csv)")
        dialog.setViewMode(QFileDialog.ViewMode.List)

        self.file_data = []
        self.file_data_non_numbers = []

        # Selection of data file
        if dialog.exec():
            filenames = dialog.selectedFiles()

            if len(filenames) >= 1:
                self.filename = filenames[0]
                i = 0

                while i < self.side_panel_layout.count():
                    if self.side_panel_layout.itemAt(i):
                        if self.side_panel_layout.itemAt(i).widget().accessibleName() in self.removable_widgets_load_data:
                            self.remove_widget(self.side_panel_layout, self.side_panel_layout.itemAt(i).widget())
                            i = i - 1

                    i = i + 1

                # Reset of graphical user interface components
                self.data_non_numbers_option_combo_box.setCurrentIndex(0)
                self.clustering_algorithm_combo_box.setCurrentIndex(0)
                self.num_clusters_input.setText("")

                # Reset data clustering display dimensionality selection
                if self.display_dimensions_button_group.checkedButton():
                    self.display_dimensions_button_group.setExclusive(False)
                    self.display_dimensions_button_group.checkedButton().setChecked(False)
                    self.display_dimensions_button_group.setExclusive(True)

                # Reset of flag variables
                self.data_imported = False
                self.file_data_error = False
                self.data_non_numbers = False
                self.feature_values_non_numbers_entered = False
                self.feature_values_non_numbers_entered_not_applicable = False
                self.clustering_algorithm_selected = False
                self.num_clusters_entered = False
                self.num_clusters_not_applicable = False
                self.dimension_selected = False

                # Loading of data file and checking for any errors
                with open(self.filename, "r") as data_file:
                    try:
                        self.file_data = np.genfromtxt(data_file, delimiter=',')

                    except ValueError as error:
                        if "columns instead of" in error.args[0]:
                            self.file_data_error_label.setText("The file must contain data points with\nthe same number of entries denoting\nthe features and class.")
                            self.insert_widget(self.side_panel_layout, self.data_with_classes_check_box, self.file_data_error_label)
                            self.file_data_error = True

                    else:
                        data_file.seek(0)

                        if not self.file_data.any():
                            self.file_data_error_label.setText("The file does not contain any data.")
                            self.insert_widget(self.side_panel_layout, self.data_with_classes_check_box, self.file_data_error_label)
                            self.file_data_error = True

                        elif len(data_file.readlines()) == 1 or len(self.file_data.shape) == 0:
                            self.file_data_error_label.setText("The file must contain at least two\ndata points.")
                            self.insert_widget(self.side_panel_layout, self.data_with_classes_check_box, self.file_data_error_label)
                            self.file_data_error = True

                        elif len(self.file_data.shape) == 1:
                            self.file_data_error_label.setText("The file must contain at least two\ndata points and two features.")
                            self.insert_widget(self.side_panel_layout, self.data_with_classes_check_box, self.file_data_error_label)
                            self.file_data_error = True

                        else:
                            self.data_imported = True
                            self.file_data_error = False
                            self.data_non_numbers = False
                            self.feature_values_non_numbers_entered = False
                            self.feature_values_non_numbers_entered_not_applicable = False
                            self.process_data()

    def process_data(self):
        """Checks if the data has enough features for data clustering to be performed.
        Checks if the data has non-numeric values and processes non-numeric values
        and includes graphical user interface components for handling non-numeric values
        if the data contains such values."""
        if self.data_imported:
            if len(self.data_non_numbers_widget.children()) > 1:
                for i in self.data_non_numbers_widget.children():
                    self.remove_widget_child(self.data_non_numbers_layout)

            i = 0

            while i < self.side_panel_layout.count():
                if self.side_panel_layout.itemAt(i):
                    if self.side_panel_layout.itemAt(i).widget().accessibleName() in self.removable_widgets_process_data:
                        self.remove_widget(self.side_panel_layout, self.side_panel_layout.itemAt(i).widget())
                        i = i - 1

                i = i + 1

            self.data_non_numbers = False

            if not self.button3D.isEnabled():
                self.button3D.setEnabled(True)

            if self.data_with_classes:
                if self.file_data.shape[1] == 2:
                    self.file_data_error_label.setText("The file must contain data points with\nan assigned class and at least two\nfeatures.")
                    self.insert_widget(self.side_panel_layout, self.data_with_classes_check_box, self.file_data_error_label)
                    self.file_data_error = True

                # Checks for non-numeric values in the data
                else:
                    for row in self.file_data:
                        if np.isnan(row[:-1]).any():
                            self.data_non_numbers = True
                            break

            else:
                self.remove_widget(self.side_panel_layout, self.file_data_error_label)
                self.file_data_error = False

                # Checks for non-numeric values in the data
                for row in self.file_data:
                    if np.isnan(row).any():
                        self.data_non_numbers = True
                        break

            # Processes the data if the data has no errors
            if not self.file_data_error:
                self.attribute_data_all = []
                self.attribute_data = []
                self.class_data = []
                self.num_classes = 0

                # Processes the data if it contains non-numeric values
                if self.data_non_numbers:
                    with open(self.filename, "r") as data_file:
                        self.file_data_non_numbers = np.genfromtxt(data_file, dtype=str, delimiter=',', filling_values=-99999, encoding='utf8')
                        self.file_data_non_numbers = np.array([list(data_point) for data_point in self.file_data_non_numbers])

                    if self.data_with_classes:
                        self.attribute_data_all = self.file_data_non_numbers[:, :-1]
                        self.class_data = self.file_data_non_numbers[:, -1]
                        self.num_classes = np.unique(self.class_data).size

                    else:
                        self.attribute_data_all = self.file_data_non_numbers

                elif self.data_with_classes:
                    self.attribute_data_all = self.file_data[:, :-1]
                    self.class_data = self.file_data[:, -1]
                    self.num_classes = np.unique(self.class_data).size
                    self.attribute_data = self.attribute_data_all
                    self.feature_values_non_numbers_entered_not_applicable = True

                else:
                    self.attribute_data_all = self.file_data
                    self.attribute_data = self.attribute_data_all
                    self.feature_values_non_numbers_entered_not_applicable = True

                self.non_number_values_file = [[] for col in range(self.attribute_data_all.shape[1])]
                self.non_number_values = []

                # Keeps track of non-numeric values so that they can be converted into numeric values for clustering of numeric data
                if self.data_non_numbers:
                    for col in range(self.attribute_data_all.shape[1]):
                        for row, value in enumerate(self.attribute_data_all[:, col]):
                            repeated_value = 0

                            if value and not value == "-99999" and not value == "-99999.0":
                                if value[0] == "-":
                                    if not str.isnumeric(value.replace('-', '', 1).replace('.', '', 1)):
                                        for feature_value in self.non_number_values_file[col]:
                                            if value == feature_value[2]:
                                                feature_value[1].append(row)
                                                repeated_value = 1
                                                break

                                        if not repeated_value:
                                            self.non_number_values_file[col].append([col, [row], value, None])

                                elif value[0] == "+":
                                    if not str.isnumeric(value.replace('+', '', 1).replace('.', '', 1)):
                                        for feature_value in self.non_number_values_file[col]:
                                            if value == feature_value[2]:
                                                feature_value[1].append(row)
                                                repeated_value = 1
                                                break

                                        if not repeated_value:
                                            self.non_number_values_file[col].append([col, [row], value, None])

                                elif not str.isnumeric(value.replace('.', '', 1)):
                                    for feature_value in self.non_number_values_file[col]:
                                        if value == feature_value[2]:
                                            feature_value[1].append(row)
                                            repeated_value = 1
                                            break

                                    if not repeated_value:
                                        self.non_number_values_file[col].append([col, [row], value, None])

                            else:
                                self.attribute_data_all[row, col] = ""

                                for feature_value in self.non_number_values_file[col]:
                                    if self.attribute_data_all[row, col] == feature_value[2]:
                                        feature_value[1].append(row)
                                        repeated_value = 1
                                        break

                                if not repeated_value:
                                    self.non_number_values_file[col].append([col, [row], self.attribute_data_all[row, col], None])

                    self.non_number_values_file = [self.non_number_values_file[feature_index] for feature_index in range(len(self.non_number_values_file)) if self.non_number_values_file[feature_index]]
                    self.non_number_values = self.non_number_values_file

                if self.data_with_classes:
                    if int(self.num_classes) > 1:
                        self.data_classes_label.setText("The selected data file has " + str(self.num_classes) + " classes.")

                    else:
                        self.data_classes_label.setText("The selected data file has " + str(self.num_classes) + " class.")

                    self.insert_widget(self.side_panel_layout, self.data_with_classes_check_box, self.data_classes_label)

                # Constructs graphical user interface components for assigning non-numeric values to numbers
                if self.data_non_numbers:
                    self.features_non_numbers_combo_box = QComboBox()
                    self.features_non_numbers_combo_box.setFixedWidth(100)
                    features_non_numbers_combo_box_list = []

                    self.feature_values_non_numbers_combo_box_list = [None] * self.attribute_data_all.shape[1]

                    self.feature_values_non_numbers_input = QLineEdit()
                    doubleInput = QDoubleValidator()
                    doubleInput.setRange(-999999999, 999999999)
                    doubleInput.setDecimals(9)
                    self.feature_values_non_numbers_input.setValidator(doubleInput)
                    self.feature_values_non_numbers_input.textChanged.connect(self.feature_values_non_numbers_changed)
                    self.feature_values_non_numbers_input.setFixedWidth(50)

                    for feature in self.non_number_values:
                        features_non_numbers_combo_box_list.append("Feature " + str(feature[0][0] + 1))

                    self.features_non_numbers_combo_box.addItems(features_non_numbers_combo_box_list)

                    for feature in self.non_number_values:
                        feature_values_non_numbers_combo_box = QComboBox()
                        feature_values_non_numbers_combo_box.setAccessibleName("Feature Values")
                        feature_values_non_numbers_combo_box.setFixedWidth(70)
                        self.feature_values_non_numbers_combo_box_list[feature[0][0]] = feature_values_non_numbers_combo_box
                        values_non_numbers_combo_box_list = []

                        for value in feature:
                            values_non_numbers_combo_box_list.append(value[2])

                        self.feature_values_non_numbers_combo_box_list[feature[0][0]].addItems(
                            values_non_numbers_combo_box_list)

                    self.feature_values_non_numbers_combo_box_list = [
                        self.feature_values_non_numbers_combo_box_list[feature_index] for feature_index in
                        range(len(self.feature_values_non_numbers_combo_box_list)) if
                        self.feature_values_non_numbers_combo_box_list[feature_index]]

                    [self.feature_values_non_numbers_combo_box_list[feature_index].currentIndexChanged.connect(self.feature_values_non_numbers_selection_changed) for feature_index in range(len(self.feature_values_non_numbers_combo_box_list))]

                    self.features_non_numbers_combo_box.currentIndexChanged.connect(self.feature_non_numbers_selection_changed)

                    self.data_non_numbers_layout.addWidget(self.features_non_numbers_combo_box)
                    self.data_non_numbers_layout.addWidget(self.feature_values_non_numbers_combo_box_list[0])
                    self.data_non_numbers_layout.addWidget(self.feature_values_non_numbers_input)

                    if self.data_with_classes:
                        self.insert_widget(self.side_panel_layout, self.data_classes_label, self.data_non_numbers_label)

                    else:
                        self.insert_widget(self.side_panel_layout, self.data_with_classes_check_box, self.data_non_numbers_label)

                    self.insert_widget(self.side_panel_layout, self.data_non_numbers_label, self.data_non_numbers_option_combo_box)
                    self.data_non_numbers_option_changed()

                # Does not allow 3D display of data clustering if the data points have only 2 features as data points must
                # have at least 3 features for 3D display of data clustering
                if 2 in self.attribute_data_all.shape:
                    if self.button3D.isChecked():
                        self.display_dimensions_button_group.setExclusive(False)
                        self.button3D.setChecked(False)
                        self.display_dimensions_button_group.setExclusive(True)
                        self.display_dimension = ""
                        self.dimension_selected = False

                    self.button3D.setEnabled(False)

            self.num_clusters_changed()
            self.display_data_clustering_enable()

    def data_with_classes_option_changed(self):
        """The condition of whether or not the data contains class assignments with each data point.
        If the data contains class assignments, then the classes are not included with the data that
        is used for performing clustering. The data is reprocessed when this option is changed."""
        if self.data_with_classes_check_box.isChecked():
            self.data_with_classes = True
            self.process_data()

        else:
            self.data_with_classes = False
            self.process_data()

    def data_non_numbers_option_changed(self):
        """The option that is selected for how to handle non-numeric data values.
        The clustering is performed on numeric data and thus these options pertain
        to different ways for converting data containing non-numeric values into
        numeric data."""

        # Empty selection option for handling non-numeric values in the data
        if self.data_non_numbers_option_combo_box.currentIndex() == 0:
            self.remove_widget(self.side_panel_layout, self.data_non_numbers_widget)
            self.feature_values_non_numbers_input.clearFocus()
            self.feature_values_non_numbers_entered_not_applicable = False
            self.feature_values_non_numbers_entered = False
            self.display_data_clustering_enable()

        # Option for setting all non-numeric values to zero
        elif self.data_non_numbers_option_combo_box.currentIndex() == 1:
            self.attribute_data = self.attribute_data_all

            for feature in self.non_number_values:
                for value in feature:
                    for row in value[1]:
                        self.attribute_data[row, value[0]] = 0

            self.attribute_data = self.attribute_data.astype(float)

            self.remove_widget(self.side_panel_layout, self.data_non_numbers_widget)

            self.feature_values_non_numbers_input.clearFocus()
            self.feature_values_non_numbers_entered_not_applicable = True
            self.feature_values_non_numbers_entered = False
            self.display_data_clustering_enable()

        # Option for assigning non-numeric values to number input
        elif self.data_non_numbers_option_combo_box.currentIndex() == 2:
            self.attribute_data = np.empty(self.attribute_data_all.shape, dtype=self.attribute_data_all.dtype)
            non_number_indices = [(row, col[0][0]) for col in self.non_number_values_file for value in col for row in value[1]]

            for index, value in np.ndenumerate(self.attribute_data_all):
                if not index in non_number_indices:
                    self.attribute_data[index] = self.attribute_data_all[index]

            for feature in self.non_number_values:
                for value in feature:
                    if value[3]:
                        for row in value[1]:
                            self.attribute_data[row, value[0]] = value[3]

                    else:
                        for row in value[1]:
                            self.attribute_data[row, value[0]] = None

            self.display_data_clustering_window1_button.setEnabled(False)
            self.display_data_clustering_window2_button.setEnabled(False)

            self.remove_widget(self.side_panel_layout, self.data_non_numbers_widget)

            feature_index = self.features_non_numbers_combo_box.currentIndex()
            feature_value_index = self.feature_values_non_numbers_combo_box_list[feature_index].currentIndex()
            self.feature_values_non_numbers_input.setText(self.non_number_values[feature_index][feature_value_index][3])

            self.data_non_numbers_layout.addWidget(self.features_non_numbers_combo_box)
            self.data_non_numbers_layout.addWidget(self.feature_values_non_numbers_combo_box_list[feature_index])
            self.data_non_numbers_layout.addWidget(self.feature_values_non_numbers_input)
            self.insert_widget(self.side_panel_layout, self.data_non_numbers_option_combo_box, self.data_non_numbers_widget)

            self.feature_values_non_numbers_entered_not_applicable = False
            feature_values_non_numbers_assigned = True

            # Check to ensure that non-numeric values have not been assigned to numbers
            for feature in self.non_number_values:
                for value in feature:
                    if not value[3]:
                        feature_values_non_numbers_assigned = False
                        break

            if feature_values_non_numbers_assigned:
                self.attribute_data = self.attribute_data.astype(float)
                self.feature_values_non_numbers_entered = True

            else:
                self.feature_values_non_numbers_entered = False

            self.display_data_clustering_enable()

        # Option for removing all rows in the data that have non-numeric values
        elif self.data_non_numbers_option_combo_box.currentIndex() == 3:
            self.attribute_data = self.attribute_data_all
            remove_rows = []

            for feature in self.non_number_values_file:
                for value in feature:
                    for row in value[1]:
                        remove_rows.append(row)

            remove_rows = np.unique(remove_rows)
            self.attribute_data = np.delete(self.attribute_data, remove_rows, 0)
            self.attribute_data = self.attribute_data.astype(float)

            self.remove_widget(self.side_panel_layout, self.data_non_numbers_widget)

            self.feature_values_non_numbers_input.clearFocus()
            self.feature_values_non_numbers_entered_not_applicable = True
            self.feature_values_non_numbers_entered = False
            self.display_data_clustering_enable()

        # Option for removing all columns in the data that have non-numeric values
        elif self.data_non_numbers_option_combo_box.currentIndex() == 4:
            self.attribute_data = self.attribute_data_all
            remove_columns = []

            for feature in self.non_number_values_file:
                remove_columns.append(feature[0][0])

            self.attribute_data = np.delete(self.attribute_data, remove_columns, 1)
            self.attribute_data = self.attribute_data.astype(float)

            self.remove_widget(self.side_panel_layout, self.data_non_numbers_widget)

            self.feature_values_non_numbers_input.clearFocus()
            self.feature_values_non_numbers_entered_not_applicable = True
            self.feature_values_non_numbers_entered = False
            self.display_data_clustering_enable()

    def feature_non_numbers_selection_changed(self):
        """Changes the non-numeric values displayed to those belonging to the selected feature for the user
        to assign numeric values to as one of the options for handling non-numeric values."""
        feature_index = self.features_non_numbers_combo_box.currentIndex()

        # Removes non-numeric values selection list from graphical user interface for the previously selected
        # data feature and adds the newly selected non-numeric values selection list for the option
        # for assigning non-numeric values to number input
        for i in range(self.data_non_numbers_layout.count()):
            if self.data_non_numbers_layout.itemAt(i).widget().accessibleName() == "Feature Values":
                replaced_widget = self.data_non_numbers_layout.itemAt(i).widget()
                self.data_non_numbers_layout.removeWidget(replaced_widget)
                replaced_widget.setParent(None)
                inserted_widget = self.feature_values_non_numbers_combo_box_list[feature_index]
                inserted_widget.setCurrentIndex(0)
                self.data_non_numbers_layout.insertWidget(i, inserted_widget)
                break

        feature_value_index = self.feature_values_non_numbers_combo_box_list[feature_index].currentIndex()
        self.feature_values_non_numbers_input.setText(self.non_number_values[feature_index][feature_value_index][3])

    def feature_values_non_numbers_selection_changed(self):
        """Changes the value for a feature to the selected value for the user to assign a numeric value
        to as one of the options for handling non-numeric values."""
        feature_index = self.features_non_numbers_combo_box.currentIndex()
        feature_value_index = self.feature_values_non_numbers_combo_box_list[feature_index].currentIndex()
        self.feature_values_non_numbers_input.setText(self.non_number_values[feature_index][feature_value_index][3])

    def feature_values_non_numbers_changed(self):
        """Updates the numeric value assigned to a non-numeric value for a feature and checks if
        all non-numeric values in the data have been assigned numeric values as a condition
        for clustering the data."""
        feature_index = self.features_non_numbers_combo_box.currentIndex()
        feature_value_index = self.feature_values_non_numbers_combo_box_list[feature_index].currentIndex()
        rows = self.non_number_values[feature_index][feature_value_index][1]
        col = self.non_number_values[feature_index][feature_value_index][0]
        feature_values_non_numbers_combo_box_text = self.feature_values_non_numbers_combo_box_list[feature_index].currentText()

        # Assigns number input to all instances of the corresponding non-numeric value if number input is already equal to
        # the numeric value for that non-numeric value to ensure non-numeric values and instances of
        # the non-numeric value in the data array are assigned to that number
        if self.non_number_values[feature_index][feature_value_index][2] == feature_values_non_numbers_combo_box_text and self.feature_values_non_numbers_input.text() and not self.feature_values_non_numbers_input.text() in self.feature_values_non_numbers_invalid_input:
            self.non_number_values[feature_index][feature_value_index][3] = self.feature_values_non_numbers_input.text()

            for row in rows:
                self.attribute_data[row][col] = self.non_number_values[feature_index][feature_value_index][3]

            self.feature_values_non_numbers_combo_box_list[feature_index].setItemText(feature_value_index, feature_values_non_numbers_combo_box_text + " (" + self.feature_values_non_numbers_input.text() + ")")

        # Assigns number input to all instances of the corresponding non-numeric value if the number input is different from
        # the numeric value assigned to the non-numeric value
        elif not self.non_number_values[feature_index][feature_value_index][2] == feature_values_non_numbers_combo_box_text and self.feature_values_non_numbers_input.text() and not self.feature_values_non_numbers_input.text() in self.feature_values_non_numbers_invalid_input:
            self.non_number_values[feature_index][feature_value_index][3] = self.feature_values_non_numbers_input.text()

            for row in rows:
                self.attribute_data[row][col] = self.non_number_values[feature_index][feature_value_index][3]

            self.feature_values_non_numbers_combo_box_list[feature_index].setItemText(feature_value_index, feature_values_non_numbers_combo_box_text.rsplit(' ', 1)[0] + " (" + self.feature_values_non_numbers_input.text() + ")")

        # Assigns non-numeric value to an empty value if there is no number input or the input is not numeric
        elif not self.non_number_values[feature_index][feature_value_index][2] == feature_values_non_numbers_combo_box_text and (not self.feature_values_non_numbers_input.text() or self.feature_values_non_numbers_input.text() in self.feature_values_non_numbers_invalid_input):
            self.non_number_values[feature_index][feature_value_index][3] = ""

            for row in rows:
                self.attribute_data[row][col] = None

            self.feature_values_non_numbers_combo_box_list[feature_index].setItemText(feature_value_index, feature_values_non_numbers_combo_box_text.rsplit(' ', 1)[0])

        feature_values_non_numbers_assigned = True

        # Checks if all non-numeric values have been assigned to a number as a condition for clustering numeric data
        for feature in self.non_number_values:
            for value in feature:
                if not value[3]:
                    feature_values_non_numbers_assigned = False
                    break

        if feature_values_non_numbers_assigned:
            self.attribute_data = self.attribute_data.astype(float)
            self.feature_values_non_numbers_entered = True

        else:
            self.feature_values_non_numbers_entered = False

        self.display_data_clustering_enable()

    def clustering_algorithm_changed(self):
        """Changes the clustering algorithm that is to be used for clustering the data to
        the selected clustering algorithm and displays or removes a component
        for entering in the number of clusters to cluster the data into if applicable to
        the algorithm."""

        # Empty selection option for clustering algorithm selection
        if self.clustering_algorithm_combo_box.currentIndex() == 0:
            self.remove_widget(self.side_panel_layout, self.num_clusters_label)
            self.remove_widget(self.side_panel_layout, self.num_clusters_input)
            self.num_clusters_input.clearFocus()
            self.clustering_algorithm_selected = False
            self.display_data_clustering_enable()

        # Clustering algorithm selection with the option to input the number of clusters to cluster the data into
        elif not self.clustering_algorithm_combo_box.currentIndex() == 0 and not self.clustering_algorithm_combo_box.currentText() in self.clustering_algorithms_no_num_clusters_input:
            self.insert_widget(self.side_panel_layout, self.clustering_algorithm_combo_box, self.num_clusters_label)
            self.insert_widget(self.side_panel_layout, self.num_clusters_label, self.num_clusters_input)
            self.clustering_algorithm_selected = True
            self.num_clusters_not_applicable = False
            self.display_data_clustering_enable()

        # Clustering algorithm selection that does not have the option to input the number of clusters to cluster the data into
        elif self.clustering_algorithm_combo_box.currentText() in self.clustering_algorithms_no_num_clusters_input:
            self.remove_widget(self.side_panel_layout, self.num_clusters_label)
            self.remove_widget(self.side_panel_layout, self.num_clusters_input)
            self.num_clusters_input.clearFocus()
            self.clustering_algorithm_selected = True
            self.num_clusters_not_applicable = True
            self.display_data_clustering_enable()

    def num_clusters_changed(self):
        """Updates the number of clusters to cluster the data into to the value entered."""
        if self.data_imported:
            if self.num_clusters_input.text() and not self.num_clusters_input.text() == "0" and int(self.num_clusters_input.text()) <= self.attribute_data_all.shape[0]:
                self.num_clusters_entered = True
                self.display_data_clustering_enable()

            else:
                self.num_clusters_entered = False
                self.display_data_clustering_enable()

    def radio_button_selected(self):
        """Checks the button selected for whether to display the graph of the clustered data in 2D or 3D."""
        radio_button = self.sender()

        if radio_button.isChecked():
            self.display_dimension = radio_button.text
            self.dimension_selected = True
            self.display_data_clustering_enable()

    def display_data_clustering_enable(self):
        """Checks if all the conditions have been met for clustering the data including
        the data being valid, all data having numeric values, and all of the clustering
        options having been selected."""
        if self.data_imported and not self.file_data_error and (not self.data_non_numbers or self.feature_values_non_numbers_entered or self.feature_values_non_numbers_entered_not_applicable) and (self.clustering_algorithm_selected and (self.num_clusters_entered or self.num_clusters_not_applicable)) and self.dimension_selected:
            self.display_data_clustering_window1_button.setEnabled(True)

            if self.display_window1 or self.display_window2:
                self.display_data_clustering_window2_button.setEnabled(True)

        else:
            self.display_data_clustering_window1_button.setEnabled(False)
            self.display_data_clustering_window2_button.setEnabled(False)

    def display_data_clustering(self):
        """Displays a graph of the data clustering in one of two display windows and in the dimensionality selected.
        If the data has class assignments, the accuracy of the algorithm's clustering classification
        is also displayed with respect to the class assignments for the data.
        Options for closing each display window and saving the clustering data and graphs are also shown."""

        # Assignment of data clustering parameters
        data_clustering_algorithm = self.clustering_algorithm_combo_box.currentText()
        num_clusters = 0 if self.num_clusters_input.text() == "" else int(self.num_clusters_input.text())
        dimension = 2 if self.display_dimension == "2D" else 3
        pca = PCA(dimension)
        pca_data = pca.fit_transform(self.attribute_data)
        title = data_clustering_algorithm
        clustering_data_labels = []
        clustering_metric = ""

        # Clustering of the data with the clustering algorithm that has been selected
        if data_clustering_algorithm == "K-Means":
            clustering_data = KMeans(num_clusters).fit(self.attribute_data)
            clustering_data_labels = clustering_data.labels_

        elif data_clustering_algorithm == "Mean Shift":
            clustering_data = MeanShift().fit(self.attribute_data)
            clustering_data_labels = clustering_data.labels_

        elif data_clustering_algorithm == "DBSCAN":
            clustering_data = DBSCAN().fit(self.attribute_data)
            clustering_data_labels = clustering_data.labels_

        elif data_clustering_algorithm == "HDBSCAN":
            clustering_data = HDBSCAN(min_samples=2).fit(self.attribute_data)
            clustering_data_labels = clustering_data.labels_

        elif data_clustering_algorithm == "Gaussian Mixture Models":
            clustering_data_labels = GaussianMixture(num_clusters).fit_predict(self.attribute_data)

        elif data_clustering_algorithm == "Agglomerative":
            clustering_data = AgglomerativeClustering(num_clusters).fit(self.attribute_data)
            clustering_data_labels = clustering_data.labels_

        elif data_clustering_algorithm == "Affinity Propagation":
            clustering_data = AffinityPropagation().fit(self.attribute_data)
            clustering_data_labels = clustering_data.labels_

        elif data_clustering_algorithm == "Spectral":
            if self.attribute_data.shape[0] < 8:
                clustering_data = SpectralClustering(self.attribute_data.shape[0]).fit(self.attribute_data)

            else:
                clustering_data = SpectralClustering().fit(self.attribute_data)

            clustering_data_labels = clustering_data.labels_

        elif data_clustering_algorithm == "BIRCH":
            clustering_data = Birch().fit(self.attribute_data)
            clustering_data_labels = clustering_data.labels_

        elif data_clustering_algorithm == "OPTICS":
            clustering_data = OPTICS(min_samples=2).fit(self.attribute_data)
            clustering_data_labels = clustering_data.labels_

        # Constructs the layout containing the graph of the data clustering
        display_window_layout = QVBoxLayout()
        display_window_widget = QWidget()
        display_window_widget.setLayout(display_window_layout)
        display_window_widget.setMinimumWidth(int((self.window_dimensions[0] - self.side_panel_widget.width() - self.separator_line.width())/3))
        separator_line_width = self.separator_line.width()
        plot_width = int(self.width() - self.side_panel_widget.width() - self.separator_line.width()) if ((self.display_window1 and not self.display_window2) or (self.display_window2 and not self.display_window1) or not any([self.display_window1, self.display_window2])) else int((self.width() - self.side_panel_widget.width() - self.separator_line.width() - separator_line_width)/2)
        plot_width_conv = plot_width / self.physicalDpiX()
        plot_height = int((17/20)*self.height())
        plot_height_conv = plot_height / self.physicalDpiY()
        data_clustering_figure = PlotWidget(plot_width_conv, plot_height_conv, title, dimension)
        data_clustering_figure.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Graphs the data clustering in 2D or 3D depending on the dimensionality that has been selected
        if dimension == 2:
            for l in np.unique(clustering_data_labels):
                data_clustering_figure.ax.scatter(pca_data[clustering_data_labels == l, 0], pca_data[clustering_data_labels == l, 1], color=plt.cm.jet(float(l)/np.max(clustering_data_labels+1)))

        else:
            for l in np.unique(clustering_data_labels):
                data_clustering_figure.ax.scatter(pca_data[clustering_data_labels == l, 0], pca_data[clustering_data_labels == l, 1], pca_data[clustering_data_labels == l, 2], color=plt.cm.jet(float(l)/np.max(clustering_data_labels+1)))

        display_window_button_layout = QHBoxLayout()
        display_window_button_widget = QWidget()
        display_window_button_widget.setLayout(display_window_button_layout)

        # Button for saving the clustering data as text or as a graph
        save_data_clustering_button = QPushButton("Save")
        save_data_clustering_button.clicked.connect(self.save_data_clustering)
        save_data_clustering_button.setFixedWidth(60)

        # Button for closing the graph display window
        close_display_window_button = QPushButton("Close")
        close_display_window_button.clicked.connect(self.close_display_window)
        close_display_window_button.setFixedWidth(60)

        display_window_button_layout.addWidget(close_display_window_button)
        display_window_button_layout.addWidget(save_data_clustering_button)
        display_window_button_layout.setContentsMargins(0, 0, 0, 0)
        display_window_button_layout.setSpacing(7)
        display_window_button_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        display_window_layout.addWidget(data_clustering_figure)

        # Adds the accuracy of the clustering algorithm's classification with respect to the class assignments in
        # the data if the data contains class assignments
        if self.data_with_classes:
            clustering_data_rand_score = rand_score(self.class_data, clustering_data_labels)
            display_window_clustering_metrics_label = QLabel("Rand index: " + "{:.2f}".format(clustering_data_rand_score))
            display_window_layout.addWidget(display_window_clustering_metrics_label)
            clustering_metric = display_window_clustering_metrics_label.text()

        else:
            display_window_layout.addSpacing(25)

        display_window_layout.addWidget(display_window_button_widget)
        display_window_layout.setContentsMargins(7, 7, 7, 7)

        # Displays the data clustering graph in the first window if the option
        # to display the data clustering in the first window is selected
        if self.sender().accessibleName() == "Show Display 1":
            self.attribute_data_window1 = self.attribute_data
            self.clustering_data_labels_window1 = clustering_data_labels
            self.data_clustering_algorithm_window1 = data_clustering_algorithm

            if self.data_clustering_algorithm_window1 not in self.clustering_algorithms_no_num_clusters_input:
                self.num_clusters_window1 = num_clusters

            else:
                self.num_clusters_window1 = None

            if self.data_with_classes:
                self.class_data_window1 = self.class_data
                self.clustering_metric_window1 = clustering_metric

            else:
                self.class_data_window1 = None
                self.clustering_metric_window1 = None

            display_window_widget.setAccessibleName("Display 1")
            data_clustering_figure.setAccessibleName("Plot Display 1")
            close_display_window_button.setAccessibleName("Close Display 1")
            save_data_clustering_button.setAccessibleName("Save Display 1")
            self.display_data_clustering_window2_button.setEnabled(True)

            if not self.display_window1:
                if self.display_window2:
                    for i in range(self.window_layout.count()):
                        if self.window_layout.itemAt(i).widget().accessibleName() == "Display 2":
                            self.separator_line_display = QFrame()
                            self.separator_line_display.setAccessibleName("Separator Line Display")
                            self.separator_line_display.setFrameShape(QFrame.Shape.VLine)
                            self.separator_line_display.setLineWidth(1)
                            separator_line_width = self.separator_line.width()

                            if self.width() <= (5/4)*self.window_dimensions[0]:
                                self.resize(self.width() + separator_line_width + plot_width, self.height())

                            elif self.width() < (17/8)*self.window_dimensions[0]:
                                self.resize(int((17/8)*self.window_dimensions[0]), self.height())

                            self.window_layout.insertWidget(i, self.separator_line_display)
                            self.window_layout.insertWidget(i, display_window_widget)
                            break

                else:
                    self.window_layout.addWidget(display_window_widget)

                self.display_window1 = True

            else:
                for i in range(self.window_layout.count()):
                    if self.window_layout.itemAt(i).widget().accessibleName() == "Display 1":
                        self.window_layout.takeAt(i).widget().deleteLater()
                        self.window_layout.insertWidget(i, display_window_widget)
                        break

        # Displays the data clustering graph in the second window if the option
        # to display the data clustering in the second window is selected
        elif self.sender().accessibleName() == "Show Display 2":
            self.attribute_data_window2 = self.attribute_data
            self.clustering_data_labels_window2 = clustering_data_labels
            self.data_clustering_algorithm_window2 = data_clustering_algorithm

            if self.data_clustering_algorithm_window2 not in self.clustering_algorithms_no_num_clusters_input:
                self.num_clusters_window2 = num_clusters

            else:
                self.num_clusters_window2 = None

            if self.data_with_classes:
                self.class_data_window2 = self.class_data
                self.clustering_metric_window2 = clustering_metric

            else:
                self.class_data_window2 = None
                self.clustering_metric_window2 = None

            display_window_widget.setAccessibleName("Display 2")
            data_clustering_figure.setAccessibleName("Plot Display 2")
            close_display_window_button.setAccessibleName("Close Display 2")
            save_data_clustering_button.setAccessibleName("Save Display 2")

            if not self.display_window2:
                self.separator_line_display = QFrame()
                self.separator_line_display.setAccessibleName("Separator Line Display")
                self.separator_line_display.setFrameShape(QFrame.Shape.VLine)
                self.separator_line_display.setLineWidth(1)
                separator_line_width = self.separator_line.width()

                if self.width() <= (5/4) * self.window_dimensions[0]:
                    self.resize(self.width() + separator_line_width + plot_width, self.height())

                elif self.width() < (17/8) * self.window_dimensions[0]:
                    self.resize(int((17/8) * self.window_dimensions[0]), self.height())

                self.window_layout.addWidget(self.separator_line_display)
                self.window_layout.addWidget(display_window_widget)
                self.display_window2 = True

            else:
                for i in range(self.window_layout.count()):
                    if self.window_layout.itemAt(i).widget().accessibleName() == "Display 2":
                        self.window_layout.takeAt(i).widget().deleteLater()
                        self.window_layout.insertWidget(i, display_window_widget)
                        break

    def close_display_window(self):
        """Closes the selected display window and decreases the size of the interface if
        a display window is closed and two graphs were being displayed."""
        if self.sender().accessibleName() == "Close Display 1":
            for i in range(self.window_layout.count()):
                if self.window_layout.itemAt(i).widget().accessibleName() == "Display 1":
                    self.window_layout.takeAt(i).widget().deleteLater()

                    if self.display_window2:
                        for j in range(self.window_layout.count()):
                            if self.window_layout.itemAt(j).widget().accessibleName() == "Separator Line Display":
                                self.window_layout.takeAt(j).widget().deleteLater()
                                break

                        separator_line_width = self.separator_line.width()
                        self.resize(int(self.side_panel_widget.width() + self.separator_line.width() + (self.width() - self.side_panel_widget.width() - self.separator_line.width() - separator_line_width)/2), self.height())

                    self.display_window1 = False

                    if not self.display_window2:
                        self.display_data_clustering_window2_button.setEnabled(False)

                    break

        elif self.sender().accessibleName() == "Close Display 2":
            for i in range(self.window_layout.count()):
                if self.window_layout.itemAt(i).widget().accessibleName() == "Display 2":
                    self.window_layout.takeAt(i).widget().deleteLater()

                    if self.display_window1:
                        for j in range(self.window_layout.count()):
                            if self.window_layout.itemAt(j).widget().accessibleName() == "Separator Line Display":
                                self.window_layout.takeAt(j).widget().deleteLater()
                                break

                        separator_line_width = self.separator_line.width()
                        self.resize(int(self.side_panel_widget.width() + self.separator_line.width() + (self.width() - self.side_panel_widget.width() - self.separator_line.width() - separator_line_width)/2), self.height())

                    self.display_window2 = False

                    if not self.display_window1:
                        self.display_data_clustering_window2_button.setEnabled(False)

                    break

    def save_data_clustering(self):
        """Saves the clustering data as text with a text or CSV file or as an image of the graph."""
        dialog = QFileDialog(self)
        filename_data = dialog.getSaveFileName(self, "Save As", os.path.dirname(os.path.realpath(__file__)), "Plain Text (.txt);; Comma Separated Values (.csv);; JPEG (.jpg);; PNG (.png)")

        if filename_data[0]:
            file_type = ""
            filename_save = ""

            if filename_data[1]:
                file_type = filename_data[1].rsplit('.', 1)[1].rsplit(')', 1)[0]
                filename_save = filename_data[0] + "." + file_type

            # Saves the clustered data as a text or CSV file if either of the two options is selected
            if file_type == "txt" or file_type == "csv":

                # Saves the clustered data from the first display window if the data from the first display window is selected to be saved
                if self.sender().accessibleName() == "Save Display 1":

                    # Saves the clustered data to the file
                    np.savetxt(filename_save, self.attribute_data_window1, fmt='%.9f', delimiter=',')
                    file_save_data_list = []
                    data_clustering_algorithm_save = ''.join(["Data clustering algorithm: ", self.data_clustering_algorithm_window1, '\n'])
                    num_clusters_save = ""
                    clustering_metric_save = ""

                    if self.num_clusters_window1:
                        num_clusters_save = ''.join(["Number of clusters: ", str(self.num_clusters_window1), '\n'])

                    if self.clustering_metric_window1:
                        clustering_metric_save = ''.join([self.clustering_metric_window1, '\n'])

                        # Removes leading and trailing zeros from the clustered data
                        with open(filename_save, "r") as file_read:
                            for i, row in enumerate(file_read):
                                row = row.strip('\r').rstrip('\n')
                                row = re.sub(r'\.0*(,)|\.0*$', r'\1', row)
                                row = re.sub(r'(\.[0-9]*[1-9]+)0+(,)|(\.[0-9]*[1-9]+)0+$', r'\1\2', row)
                                row = row + "," + str(self.class_data_window1[i])
                                row = row + "," + str(self.clustering_data_labels_window1[i])
                                file_save_data_list.append(row)

                        file_save_data = '\n'.join(file_save_data_list)
                        file_save_data = ''.join([file_save_data, '\n'])

                        with open(filename_save, "w") as file_save:
                            file_save.write(file_save_data)
                            file_save.write(data_clustering_algorithm_save)

                            if num_clusters_save:
                                file_save.write(num_clusters_save)

                            file_save.write(clustering_metric_save)

                    else:

                        # Removes leading and trailing zeros from the clustered data
                        with open(filename_save, "r") as file_read:
                            for i, row in enumerate(file_read):
                                row = row.rstrip('\r').rstrip('\n')
                                row = re.sub(r'\.0*(,)|\.0*$', r'\1', row)
                                row = re.sub(r'(\.[0-9]*[1-9]+)0+(,)|(\.[0-9]*[1-9]+)0+$', r'\1\2', row)
                                row = row + "," + str(self.clustering_data_labels_window1[i])
                                file_save_data_list.append(row)

                        file_save_data = '\n'.join(file_save_data_list)
                        file_save_data = ''.join([file_save_data, '\n'])

                        with open(filename_save, "w") as file_save:
                            file_save.write(file_save_data)
                            file_save.write(data_clustering_algorithm_save)

                            if num_clusters_save:
                                file_save.write(num_clusters_save)

                # Saves the clustered data from the second display window if the data from the second display window is selected to be saved
                elif self.sender().accessibleName() == "Save Display 2":

                    # Saves the clustered data to the file
                    np.savetxt(filename_save, self.attribute_data_window2, fmt='%.9f', delimiter=',')
                    file_save_data_list = []
                    data_clustering_algorithm_save = ''.join(["Data clustering algorithm: ", self.data_clustering_algorithm_window2, '\n'])
                    num_clusters_save = ""
                    clustering_metric_save = ""

                    if self.num_clusters_window2:
                        num_clusters_save = ''.join(["Number of clusters: ", str(self.num_clusters_window2), '\n'])

                    if self.clustering_metric_window2:
                        clustering_metric_save = ''.join([self.clustering_metric_window2, '\n'])

                        # Removes leading and trailing zeros from the clustered data
                        with open(filename_save, "r") as file_read:
                            for i, row in enumerate(file_read):
                                row = row.rstrip('\r').rstrip('\n')
                                row = re.sub(r'\.0*(,)|\.0*$', r'\1', row)
                                row = re.sub(r'(\.[0-9]*[1-9]+)0+(,)|(\.[0-9]*[1-9]+)0+$', r'\1\2', row)
                                row = row + "," + str(self.class_data_window2[i])
                                row = row + "," + str(self.clustering_data_labels_window2[i])
                                file_save_data_list.append(row)

                        file_save_data = '\n'.join(file_save_data_list)
                        file_save_data = ''.join([file_save_data, '\n'])

                        with open(filename_save, "w") as file_save:
                            file_save.write(file_save_data)
                            file_save.write(data_clustering_algorithm_save)

                            if num_clusters_save:
                                file_save.write(num_clusters_save)

                            file_save.write(clustering_metric_save)

                    else:

                        # Removes leading and trailing zeros from the clustered data
                        with open(filename_save, "r") as file_read:
                            for i, row in enumerate(file_read):
                                row = row.rstrip('\r').rstrip('\n')
                                row = re.sub(r'\.0*(,)|\.0*$', r'\1', row)
                                row = re.sub(r'(\.[0-9]*[1-9]+)0+(,)|(\.[0-9]*[1-9]+)0+$', r'\1\2', row)
                                row = row + "," + str(self.clustering_data_labels_window2[i])
                                file_save_data_list.append(row)

                        file_save_data = '\n'.join(file_save_data_list)
                        file_save_data = ''.join([file_save_data, '\n'])

                        with open(filename_save, "w") as file_save:
                            file_save.write(file_save_data)
                            file_save.write(data_clustering_algorithm_save)

                            if num_clusters_save:
                                file_save.write(num_clusters_save)

            # Saves an image of the data clustering graph if the option is selected
            elif file_type == "jpg" or file_type == "png":

                # Saves an image of the data clustering graph from the first display window if the option to save the graph from the first display window is selected
                if self.sender().accessibleName() == "Save Display 1":
                    for i in range(self.window_layout.count()):
                        if self.window_layout.itemAt(i):
                            if self.window_layout.itemAt(i).widget().accessibleName() == "Display 1":
                                for j in range(self.window_layout.itemAt(i).widget().layout().count()):
                                    if self.window_layout.itemAt(i).widget().layout().itemAt(j):
                                        if self.window_layout.itemAt(i).widget().layout().itemAt(j).widget().accessibleName() == "Plot Display 1":
                                            self.window_layout.itemAt(i).widget().layout().itemAt(j).widget().fig.savefig(filename_save)
                                            break

                                break

                # Saves an image of the data clustering graph from the second display window if the option to save the graph from the second display window is selected
                elif self.sender().accessibleName() == "Save Display 2":
                    for i in range(self.window_layout.count()):
                        if self.window_layout.itemAt(i):
                            if self.window_layout.itemAt(i).widget().accessibleName() == "Display 2":
                                for j in range(self.window_layout.itemAt(i).widget().layout().count()):
                                    if self.window_layout.itemAt(i).widget().layout().itemAt(j):
                                        if self.window_layout.itemAt(i).widget().layout().itemAt(j).widget().accessibleName() == "Plot Display 2":
                                            self.window_layout.itemAt(i).widget().layout().itemAt(j).widget().fig.savefig(filename_save)
                                            break

                                break

    def insert_widget(self, widget_layout, preceding_widget, inserted_widget):
        """Inserts a graphical user interface component after the specified component."""
        for i in range(widget_layout.count()):
            if widget_layout.itemAt(i).widget().accessibleName() == preceding_widget.accessibleName():
                widget_layout.insertWidget(i + 1, inserted_widget)

    def remove_widget(self, widget_layout, removed_widget):
        """Removes a graphical user interface component and all of its subcomponents."""
        for i in range(widget_layout.count()):
            if widget_layout.itemAt(i):
                if widget_layout.itemAt(i).widget().accessibleName() == removed_widget.accessibleName():
                    if widget_layout.itemAt(i).widget().layout() and widget_layout.itemAt(i).widget().children():
                        for j in range(len(widget_layout.itemAt(i).widget().children())):
                            self.remove_widget_child(widget_layout.itemAt(i).widget().layout())

                        widget = widget_layout.itemAt(i).widget()
                        widget_layout.removeWidget(widget)
                        widget.setParent(None)

                    else:
                        widget = widget_layout.itemAt(i).widget()
                        widget_layout.removeWidget(widget)
                        widget.setParent(None)

                    break

    def remove_widget_child(self, widget_layout):
        """Removes the graphical user interface components of a graphical user interface layout."""
        if widget_layout.itemAt(0):
            if widget_layout.itemAt(0).widget().layout() and widget_layout.itemAt(0).widget().children():
                for i in range(len(widget_layout.itemAt(0).widget().children())):
                    self.remove_widget_child(widget_layout.itemAt(0).layout())

                widget = widget_layout.itemAt(0).widget()
                widget_layout.removeWidget(widget)
                widget.setParent(None)

            else:
                widget = widget_layout.itemAt(0).widget()
                widget_layout.removeWidget(widget)
                widget.setParent(None)

class PlotWidget(FigureCanvasQTAgg):
    """Initializes the data clustering graph."""
    def __init__(self, width, height, title, dimension):
        """Initializes the data clustering graph with the specified parameters."""
        self.fig = Figure(figsize=(width, height))
        self.fig.suptitle(title)

        if dimension == 2:
            self.ax = self.fig.add_subplot(111)

        else:
            self.ax = self.fig.add_subplot(111, projection='3d')

        super(PlotWidget, self).__init__(self.fig)

if __name__ == '__main__':
    application = QApplication(sys.argv)
    interface = DataClusteringVisualizerInterface()
    interface.show()
    sys.exit(application.exec())
