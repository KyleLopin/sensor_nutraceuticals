# Sensor Nutraceuticals: Spectral Analysis of Tomato and Mango
## Overview
This project aims to develop a low-cost, non-destructive system for estimating nutraceutical compounds in tomatoes and mangoes. Utilizing spectral data from various sensors, the system predicts concentrations of:

 - Tomatoes: Lycopene and β-carotene

 - Mangoes: β-carotene and phenolic compounds

The project leverages machine learning models trained on spectral data collected from the following sensors:

 - AMS AS7262 (Visible Spectrum)

 - AMS AS7263 (Near-Infrared Spectrum)

 - AMS AS7265x (18-channel VIS to NIR Spectrum)

 - Hamamatsu C12880MA (Miniature Spectrometer)

<p float="left"> <img src="https://www.sparkfun.com/media/catalog/product/cache/a793f13fd3d678cea13d28206895ba0c/1/4/14347-01.jpg" width="200" alt="AS7262 Breakout"> 
  <img src="https://www.sparkfun.com/media/catalog/product/cache/a793f13fd3d678cea13d28206895ba0c/1/4/14351-01.jpg" width="200" alt="AS7263 Breakout"> 
  <img src="https://www.sparkfun.com/media/catalog/product/cache/a793f13fd3d678cea13d28206895ba0c/1/5/15050-SparkFun_Triad_Spectroscopy_Sensor_-_AS7265x__Qwiic_-01.jpg" width="200" alt="AS7265x Breakout"> 
  <img src="https://global.discourse-cdn.com/digikey/original/3X/b/1/b10d7f622480a8d68d773686eb573e1de2bd53d5.jpeg" width="200" alt="C12880MA Breakout"> </p>

## Hardware Components

### AMS AS7262
- **Type**: 6-channel visible light sensor  
- **Wavelengths**: 450nm, 500nm, 550nm, 570nm, 600nm, 650nm  
- **Features**: I²C interface, on-board temperature sensor, LED drivers  
- **Datasheet**: [AS7262 Datasheet](https://cdn.sparkfun.com/assets/f/b/c/c/f/AS7262.pdf)

---

### AMS AS7263
- **Type**: 6-channel near-infrared sensor  
- **Wavelengths**: 610nm, 680nm, 730nm, 760nm, 810nm, 860nm  
- **Features**: I²C interface, LED drivers  
- **Datasheet**: [AS7263 Datasheet](https://cdn.sparkfun.com/assets/1/b/7/3/b/AS7263.pdf)

---

### AMS AS7265x
- **Type**: 18-channel VIS to NIR spectral sensor  
- **Wavelengths**: 410nm to 940nm  
- **Composition**: Combines AS72651, AS72652, and AS72653 sensors  
- **Features**: I²C interface, requires firmware loading  
- **Datasheet**: [AS7265x Datasheet](https://cdn.sparkfun.com/assets/c/2/9/0/a/AS7265x_Datasheet.pdf)

---

### Hamamatsu C12880MA
- **Type**: Miniature spectrometer  
- **Wavelength Range**: 340nm to 850nm  
- **Resolution**: 15nm FWHM  
- **Features**: CMOS linear image sensor, 288 pixels  
- **Datasheet**: [C12880MA Datasheet](https://www.hamamatsu.com/eu/en/product/optical-sensors/spectrometers/mini-spectrometer/C12880MA.html)
