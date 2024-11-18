<div align="center" markdown>

<img src="xxx"/>

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

**Step 1:** Run the application from Ecosystem

IMAGE_HERE<br><br>

**Step 2:** Open Image Labeling Toolbox<br><br>

**Step 3:** Select `Apps` tab, open the Application

IMAGE_HERE<br><br>

**Step 4:** Select image/bounding box of interest, and click `MATCH BBOXES` button

IMAGE_HERE<br><br>

After finishing using the app, don't forget to stop the app session manually in the App Sessions.

## Acknowledgements

This app is based on the great work [LightGlue](https://github.com/cvg/LightGlue)