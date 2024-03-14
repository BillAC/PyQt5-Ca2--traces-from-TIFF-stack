# Ca2+ traces from a TIFF stack

This Python code is designed to analyze Ca2+ time course traces of cells imaged with a microscope. The cells are loaded with a Ca2+-sensitive fluorescent indicator, such as Fluo-4/AM. Images are recorded in stacked TIFF format. That is, a single TIFF file that contains multiple images acquired over time.

## How to use the program

A sample TIFF stack is provided for testing purposes (Glass_Pin_small.tif). For space saving, the stack was significantly downsized in terms of the number of images in the stack and the height/width dimensions.

**NOTE** that the sample image was recorded with both a brightfield and fluorescence image at each time point. The code therefore analyzes every second image, starting with the first. This can be changed in the code in “def open_file(self)”. 

The Python code will read the image stack (File -\> Open) and display the first image in the stack. One can play the image as a movie (Process -\> Play).

### Automated analysis

Ca2+ traces can be calculated automatically (Process -\> Automatic), which will attempt to find the cells within the image from the maximum projection of all the images in the stack. The identified regions-of-interest (ROI) will be used to calculate the average fluorescence intensity within each ROI. The F/F0 values will be calculated for each cell and displayed as a function of time. Automated peak detection is performed and the following results are displayed: The cell for which calculations are made, the image\# in the stack where the maximum occurred, the F/F0 peak amplitude and peak width. The calculated data can be exported as an Excel file (File -\> Save as).

### Manual analysis

The manual analysis (Process -\> Step-by-step), does much of the same, but allows the user to change some of the default parameters, such as image interval (default is 5 s), minimum cell size (default is 50 pixels), and maximum cell size (default is 6000 pixels). These values can also be directly changed in the code: class MainWindow(QMainWindow). Additionally, with manual processing more of the underlying steps are displayed.

# License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see \<http://www.gnu.org/licenses/\>.
