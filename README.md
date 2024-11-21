<div align="center" markdown>

<img src="https://github.com/user-attachments/assets/5e7e931e-376c-4066-baaf-2b750e000649"/>

# Multiview Object Matching

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/multiview-object-matching)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/multiview-object-matching)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/multiview-object-matching.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/multiview-object-matching.png)](https://supervisely.com)

</div>

## Overview

**Multiview Object Matching** is an image labeling toolbox application made for matching bounding boxes across multiview images. App works both from selected bounding box object, and from all bounding boxes on an image. Once the image/bbox is selected and `MATCH BBOXES` button is pressed, all other images in the multiview group will be updated with matched bounding boxes.

## LightGlue

This application uses [LightGlue](https://github.com/cvg/LightGlue) (ICCV 2023). A lightweight neural network designed for image feature matching. It is used in conjunction with a feature descriptor, specifically, **SuperPoint**. LightGlue and SuperPoint form a powerful pipeline for various computer vision tasks such as image matching and localization. It achieves state-of-the-art performance on several benchmarks while being faster than previous methods.

## How To Run

**Step 1:** Open Image Labeling Toolbox, select `Apps` tab and run the Application

<img src="https://github.com/user-attachments/assets/5d379f64-eadc-44ad-b137-e9bd0fd3967e"/><br><br>

**Step 2:** Select image/bounding box of interest<br><br>

**Step 3: Optional** Configure processing device, and LightGlue settings

**Step 4:** Click `MATCH BBOXES` button

<img src="https://github.com/user-attachments/assets/53a804a5-9185-4800-99ed-4f5153f88068"/><br><br>

After finishing using the app, don't forget to stop the app session manually in the App Sessions.

## Acknowledgements

This app is based on the great work [LightGlue](https://github.com/cvg/LightGlue)![GitHub Org's stars](https://img.shields.io/github/stars/cvg/LightGlue?style=social)