# !/usr/bin/env python
# =============================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Created on Fri Jul 6 11:50:00 2023
# @author: William A. Coetzee
# %  Code to:
# %    - Open a TIFF stack image containing time-lapse
# %      flouresence data 
# %    - automatically correct background (optional)
# %    - automatically select ROI's in the images
# %    - perform particle analysis
# %    - save data to an excel file
# %
# %    Reading:
# %    https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
# %    http://jkimmel.net/so-you-want-to-segment-a-cell/
# %    https://scikit-image.org/docs/stable/api/skimage.io.html
# %    https://stackoverflow.com/questions/32886239/python-image-analysis-reading-a-multidimensional-tiff-file-from-confocal-micros
# %
# %   Version 1.3 (7/6/2023)
# %   - Changed the code
# %   - Implement as a Windows GUI using PyQt5
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# =============================================================================
# Notes
# Anaconda3 and Spyder live in "C:\Users\coetzw01\anaconda3"
# Anaconda > Environments base (root) > update all packages
# Install package in Anaconda Navigator > Environments

import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QMessageBox, QLabel, QVBoxLayout,  QDesktopWidget
from PyQt5.QtWidgets import QDialog, QLineEdit, QDialogButtonBox
from PyQt5.QtCore import Qt
from skimage import io as skio
from skimage import filters as skfilt, feature; 
from scipy import ndimage as ndi;
from scipy import signal;
import numpy as np
import pandas as pd
import os;

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches


__debugging__ = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Image Processing App")
        self.setGeometry(100, 100, 1000, 1000)
        self.setAcceptDrops(True)
        
        # create menus
        self.create_menus()
        
        # create table object
        self.table = QtWidgets.QTableView()
        
        # create matlabplot canvas 
        self.figure = Figure(figsize=(5, 3), facecolor="white", layout="tight")
        self.canvas = FigureCanvas(self.figure)
        self.setCentralWidget(self.canvas)
        self.center_window()
        
        
        # variables
        self.image_filename = None
        self.image_stack = None
        self.img_interval = 5
        self.min_cell_size = 50
        self.max_cell_size = 6000
        self.Auto = False
        
        self.F_F0 = None
        self.df_peaks = None
    
      
    def center_window(self):
        frame_geometry = self.frameGeometry()
        screen_center = QDesktopWidget().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())
    
    def create_menus(self):

        # Create File menu
        file_menu = self.menuBar().addMenu("File")

        filenew_action = QAction("New", self)
        filenew_action.setShortcut("Ctrl+N")
        filenew_action.triggered.connect(self.file_new)
        filenew_action.setStatusTip('Clear all variables')
        file_menu.addAction(filenew_action)

        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        open_action.setStatusTip('Open a TIF image stack')
        file_menu.addAction(open_action)


        save_action = QAction("Save as...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setStatusTip('Save File')
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Create Process menu
        process_menu = self.menuBar().addMenu("Process")

        automatic_action = QAction("Play", self)
        automatic_action.triggered.connect(self.process_play)
        process_menu.addAction(automatic_action)

        automatic_action = QAction("Automatic", self)
        automatic_action.triggered.connect(self.process_automatic)
        process_menu.addAction(automatic_action)

        step_by_step_action = QAction("Step-by-step", self)
        step_by_step_action.triggered.connect(self.process_step_by_step)
        process_menu.addAction(step_by_step_action)

        # Create Help menu
        help_menu = self.menuBar().addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)


    def file_new(self): 
        self.statusBar().showMessage('Ready to process a new file')
        
        self.image_filename = None
        self.image_stack = None
        self.img_interval = 5
        self.min_cell_size = 50
        self.max_cell_size = 6000
        self.Auto = False
        
        self.F_F0 = None
        self.df_peaks = None
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_axis_off()
        self.canvas.draw()
        QApplication.processEvents()


    def open_file(self): 
        self.statusBar().showMessage('Loading image file ...')
        
        file_dialog = QFileDialog()
        file_name, _ = file_dialog.getOpenFileName(self, "Open TIFF File", "", "TIFF Files (*.tiff *.tif)")
        
        if file_name:
            self.image_filename = file_name
            self.image_stack = skio.imread(file_name)
        
        # take every second image, starting with the first
        self.image_stack = self.image_stack[::2]; 
        self.display_first_image()
        
        self.statusBar().clearMessage()
    
    
    def save_file(self):
        self.statusBar().showMessage('Saving the calculated data ...')
        
        if self.image_filename is None:
            QMessageBox.information(self, "No file loaded", "First load an image before proceeding")
            self.statusBar().clearMessage()
            return
        pre, ext = os.path.splitext(self.image_filename);
        file_out = pre + '.xlsx';
        
        if __debugging__:
            QMessageBox.information(self, "File name", f"Suggested file name: {file_out}")
        
        # check that data are present
        if self.F_F0 is None:
            QMessageBox.information(self, "No data", "First calulate the results before saving them")
            self.statusBar().clearMessage()
            return
        
        # Save the data to an Excel spreadsheet
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, caption="Save Excel Spreadsheet", 
                                                   directory=file_out,
                                                   filter="Excel Files (*.xlsx)")
        
        with pd.ExcelWriter(file_path) as writer:
            self.F_F0.to_excel(writer, sheet_name='F_F0', header=True)
            self.df_peaks.to_excel(writer, sheet_name='Peaks', header=True)
        
        if file_path:
            # Save the processed Excel spreadsheet here
            QMessageBox.information(self, "File Saved", f"Saved file: {file_path}")
        
        self.statusBar().clearMessage()
    
    
    def display_first_image(self):
        if self.image_stack is not None:      
            
            # print to console
            print(self.image_filename) 
            print('Image stack shape is: %d %d %d' % (np.shape(self.image_stack)))
            print("dtype of image: {}".format(self.image_stack.dtype))
            bit_depth = np.iinfo(self.image_stack.dtype).max
            print('Image bit depth is: %d ' % bit_depth)
            
            # Display first image
            self.display_image(self.image_stack[0,], "The first image in the stack")
           
            # Display file info in the GUI
            msg = self.image_filename  \
                + "\n\nThere are " + str(self.image_stack.shape[0]) + " images in the stack" \
                + "\n Each image is " + str(self.image_stack.shape[1]) \
                + " X " + str(self.image_stack.shape[2]) + " pixels" \
                    + "\nEach pixel has " + str(f"{bit_depth:,}") + " grayscale levels"
            QMessageBox.information(self, "first image", msg)
            
        else:
            QMessageBox.information(self, "File error", "No image was loaded...")

    def display_image(self, image, title_text):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(image, cmap='inferno')
        ax.set_axis_off()
        ax.set_title(title_text, fontdict={'fontsize':20}, color='black')
        self.canvas.draw()
        QApplication.processEvents()


    def process_play(self):
        self.statusBar().showMessage('Displaying all images in the stack ...')
        
        # Display the sequence of images in sucession (as in a movie)
        if self.image_stack is None:
            self.open_file()
        
        num_images = self.image_stack.shape[0];    
        index=0    
        for image in self.image_stack:
            self.display_image(image, f"Image {index+1} of {num_images}")
            if __debugging__:
                print(1+index)
            index=index+1
        QMessageBox.information(self, "Display all images", "Use the Process menu for calculations")
        
        self.statusBar().clearMessage()

    def process_step_by_step(self):
        # Open the image stack if not already open 
        if self.image_stack is None:
            self.open_file()
        
        num_images = self.image_stack.shape[0];
        
        # Obtain user input for processing parameters
        if not self.Auto:
            parameters = ProcessingDialog.get_parameters(self)
            if parameters is not None:
                self.img_interval = parameters["img_interval"]
                self.min_cell_size = parameters["min_cell_size"]
                self.max_cell_size = parameters["max_cell_size"]
            
        #print(self.img_interval)
        
        # Calculate the max projection of all the images in the stack
        max_projection = np.max(self.image_stack, axis=0)
        
        self.display_image(max_projection, "Maximum projection")
        self.statusBar().showMessage('This is the maximum projection of all the images in the stack')
        QMessageBox.information(self, "Image Processing", "This is the maximum projection of all the images in the stack\n\nNow finding the cells in the image")
        self.statusBar().clearMessage()


        # Image thresholding to find the cells in the max projection
        threshold = skfilt.threshold_li(image=max_projection, initial_guess=np.quantile(max_projection, 0.5)); # threshold
        mask_li = max_projection > threshold;
        
        if not self.Auto:
            self.display_image(mask_li, "Li threshold mask")
            QMessageBox.information(self, "Image Processing", "Image thresholding to find the cells in the max projection")

        Icells = np.where(mask_li==1, max_projection, mask_li)
        #Iback  = np.where(mask_li==0, max_projection, mask_li)
        #background_value = np.median(Iback)

        # Edge detection
        edges_canny = feature.canny(image = Icells, 
                              low_threshold = np.quantile(max_projection, 0.25),
                              high_threshold= np.quantile(max_projection, 0.75), 
                              sigma=1.2); 
        if __debugging__:
            self.display_image(edges_canny, "canny edges")
            QMessageBox.information(self, "Image Processing", "Edge detection using the Canny method")

        # close the gaps        
        edges = ndi.binary_closing(edges_canny)  
        
        if not self.Auto:
            self.display_image(edges_canny, "Canny edges after closing gaps")
            QMessageBox.information(self, "Image Processing", "Canny edges after closing gaps")

        # find cells
        cells_detected = ndi.binary_fill_holes(edges).astype(int)
        
        if __debugging__:
            self.display_image(cells_detected, "Detected cells")
            QMessageBox.information(self, "Image Processing", "Detected cells. Now labeling them")

        # label the cells
        s = [[1,1,1],
             [1,1,1],
             [1,1,1]]
        labeled_array, nb_labels = ndi.label(mask_li, structure=s);
        labels = labeled_array.ravel();  # Returns a contiguous flattened array
        sizes = np.bincount(labels); #Count number of occurrences of each value in array of non-negative ints.
        print('Number of cells detected: ' + str(nb_labels))
        
        # plot the labeled objects 
        if not self.Auto:
            self.display_image(labeled_array, "labeled objects")
            
            
        # Display detected cells  in the GUI
            self.statusBar().showMessage('Cells detected')
            msg = self.image_filename  \
                + "\n\nNumber of cells detected: " + str(nb_labels) \
                + "\nEach detected cell is displayed in in a different color" 
            QMessageBox.information(self, "Detected objects. Now eliminating small and large objects", msg)
            self.statusBar().clearMessage()
            
        
        # remove bad data (small or large cells)
        mask_sizes = (sizes > self.min_cell_size) & (sizes < self.max_cell_size) ;
        mask_cells = mask_sizes[labeled_array];

        # re-detect the cells after elimating bad data
        labeled_array, nb_labels = ndi.label(mask_cells, structure=s)
        locations = ndi.find_objects(labeled_array)
        
        print('After data correction, number of cells detected: ' + str(nb_labels))
        
        # plot the labeled objects 
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(labeled_array, cmap='inferno')
        
        # Loop through each labeled object and add a text label and a bounding box
        for obj_label, bbox in enumerate(locations, start=1):
            obj_center = ((bbox[1].start + bbox[1].stop) // 2, (bbox[0].start + bbox[0].stop) // 2)
            
            # Add a text label with the object number at its center
            ax.text(*obj_center, str(obj_label), color='white', ha='center', va='center', fontsize=20, weight='bold')
        
            # Add a bounding box around the object
            rect = mpatches.Rectangle((bbox[1].start, bbox[0].start), bbox[1].stop - bbox[1].start, bbox[0].stop - bbox[0].start,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        
        ax.set_axis_off()
        ax.set_title("Cells detected", fontdict={'fontsize':20}, color='black')
        self.canvas.draw()
        QApplication.processEvents()

        
        #QMessageBox.information(self, "Image Processing", "Labeled objects after removing small and large cells")
  
        self.statusBar().showMessage('Cells detected')
        msg = self.image_filename  \
            + "\n\nNumber of cells detected: " + str(nb_labels) \
            + "\nEach detected cell is displayed in in a different color" \
            + "\n\nNow calculating the fluoresence for each cell in each image. This may take a few seconds"
        QMessageBox.information(self, "Detected cells", msg)
        #self.statusBar().clearMessage()  

        #### Calculate the average fluoresence intensity for each cell in all the images
        self.statusBar().showMessage('Calculating ...')
        QApplication.processEvents()
        
        F = np.zeros((num_images, nb_labels))
        for n in range(num_images):
            for i in range(nb_labels):
                F[n,i] = ndi.mean(self.image_stack[n,], labels=labeled_array, index=i)

        # get the fluoresence data as a function of time
        t = np.arange(0., num_images)
        t = (t*self.img_interval)
        df = pd.DataFrame(F, index=t)

        # Calculate F/F0 
        self.F_F0 = df/df.iloc[0];

        self.statusBar().clearMessage()

        if __debugging__:
            print(self.F_F0)
        
        # Display the fluoresence traces as a function of time 
        self.statusBar().showMessage('Fluoresence (F/F0) for each cell over time ...')
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(t, self.F_F0)
        ax.set_title("F/F0", fontdict={'fontsize':20}, color='black')
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Flouresence (F/F0)', fontsize=20)
        self.canvas.draw()
        QApplication.processEvents()

        # Peak detection 
        QMessageBox.information(self, "Manual Processing", "Now performing peak detection")
        k=0
        self.df_peaks = pd.DataFrame(columns=['Cell', 'Image', 'Amplitude', 'Width'])
        for n in range(nb_labels):
            img_num, peak_properties = signal.find_peaks(self.F_F0[n], height=5.0, width=1, distance=12)
            if len(img_num) > 0:
                j=0
                for i in img_num:
                    a = peak_properties["peak_heights"][j]
                    b = peak_properties["widths"][j]
                    self.df_peaks.loc[k]= (n, img_num[j], a, b)
                    j = j+1
                    k = k+1
        self.df_peaks['Cell'] = self.df_peaks['Cell'].astype('int')
        self.df_peaks['Image'] = self.df_peaks['Image'].astype('int')
        
        if __debugging__:
            print(self.df_peaks)
        
        # Display the peak values as a table
        self.model = PandasModel(self.df_peaks)
        self.table.setModel(self.model)
        self.setCentralWidget(self.table)
        self.statusBar().showMessage('Detected peaks. Remember to save the data to an Excel file')

    def process_automatic(self):
        self.Auto = True
        self.process_step_by_step()
        
        
    def show_about(self):
        QMessageBox.about(self, "About", "Image Processing App\nVersion 1.0")  
    
    
    
class ProcessingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Automatic Processing")
        self.layout = QVBoxLayout()

        # Create labels and line edits for user input
        self.img_interval_label = QLabel("Image Interval (seconds; default = 5):")
        self.img_interval_edit = QLineEdit()
        self.min_cell_size_label = QLabel("Minimum Cell Size (pixels; default = 50):")
        self.min_cell_size_edit = QLineEdit()
        self.max_cell_size_label = QLabel("Maximum Cell Size (pixels; default = 6000):")
        self.max_cell_size_edit = QLineEdit()

        self.layout.addWidget(self.img_interval_label)
        self.layout.addWidget(self.img_interval_edit)
        self.layout.addWidget(self.min_cell_size_label)
        self.layout.addWidget(self.min_cell_size_edit)
        self.layout.addWidget(self.max_cell_size_label)
        self.layout.addWidget(self.max_cell_size_edit)

        # Create OK and Cancel buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.layout.addWidget(self.buttons)
        self.setLayout(self.layout)

    def get_input_values(self):
        return {
            "img_interval": float(self.img_interval_edit.text() or "5"),
            "min_cell_size": int(self.min_cell_size_edit.text() or "50"),
            "max_cell_size": int(self.max_cell_size_edit.text() or "6000"),
        }

    @staticmethod
    def get_parameters(parent=None):
        dialog = ProcessingDialog(parent)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.get_input_values()
        return None



class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(PandasModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            
            if isinstance(value, float):
                return "%.2f" % value
            
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())





