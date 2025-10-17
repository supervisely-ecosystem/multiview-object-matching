<div align="center" markdown>

<img src="https://github.com/user-attachments/assets/5e7e931e-376c-4066-baaf-2b750e000649"/>

# Multiview Object Matching

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#usage-example-match-all-boxes-at-once">Usage Example</a> •
  <a href="#prepare-multi-view-images-project">Prepare Multi-view Project</a> •
  <a href="#acknowledgements">Acknowledgements</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/multiview-object-matching)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/multiview-object-matching)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/multiview-object-matching.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/multiview-object-matching.png)](https://supervisely.com)

</div>

## Overview

**Multiview Object Matching** is an image labeling toolbox application designed to match/track/auto-label bounding box objects across multiview image groups. This application streamlines the labeling process for groups of images containing identical objects, possibly captured from similar angles or perspectives. 

Once the `MATCH BBOXES` button is pressed, the app utilizes LightGlue, a state-of-the-art feature matching deep neural network, to update all images in the multiview group with the matched bounding boxes. After LightGlue is applied, keypoints are transposed to other images, generating bounding boxes. This ensures accurate and consistent object tracking across images taken from different perspectives. The application works both by selecting specific bounding box or all bounding boxes in an image.


## How To Use

**Step 0:** Application only works with Multi-view images projects. If you don't have one, check out <a href="#prepare-multi-view-images-project">this section</a> 

**Step 1:** Open Image Labeling Toolbox, select `Apps` tab and run the Application

<img src="https://github.com/user-attachments/assets/99119fa9-3710-47bb-9d83-8c4e5dfdd7ef"/><br><br>

**Step 2:** Select entire image (with bounding box objects) or a single bounding box of interest

<img src="https://github.com/user-attachments/assets/ab8ccddf-7c17-485c-b03a-92a13b9e9246"/><br><br>

**Step 3 (Optional):** Select processing device, and configure Advanced settings

<img src="https://github.com/user-attachments/assets/bcdff634-bd3a-4426-bbe6-0bc52ed526c7"/><br><br>

**Step 4:** Click `MATCH BBOXES` button

<img src="https://github.com/user-attachments/assets/9094dc44-1494-4348-997b-ab8b5dd56103"/><br><br>

After finishing using the app, don't forget to stop the app session manually in the App Sessions.

## Usage example: Match all boxes at once

![gif](https://github.com/user-attachments/assets/d37a5f00-8afb-44c1-a950-1f3c580563f6)

## Prepare Multi-view images project

There is a couple of ways you could get Multiview project:

### Option 1: Create a new project

When creating a new project, select this option:

<img src="https://github.com/user-attachments/assets/7830c806-1f82-4cbd-93ba-c335c61324ab" width=80%/>

After that, any images that you import will be grouped for multi-view labeling.

### Option 2: Get a sample project

Get a small sample project from ecosystem: <br>

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/matcher-sample" src="https://github.com/user-attachments/assets/c1531f84-41e7-43a2-abb3-2f9988044bc8" height="100px" margin-bottom="20px"/>

### Option 3: "Group Images for Multiview Labeling" Application

You could run [this application](https://ecosystem.supervisely.com/apps/group-images-for-multiview-labeling) on your existing project to group images for multi-view labeling.
Application allows to group images by tags, instances of object classes, or simply by batches (just a number of images).

<img src="https://github.com/user-attachments/assets/8c983c14-ab70-46ce-ab29-fb930a6e7864"/>

### Option 4: Import Multi-view Project

Multi-view images projects could also be imported via [Import images groups](https://ecosystem.supervisely.com/ecosystem/apps/import-images-groups) application. Just drag & drop the archive. If you don't have a project at your disposal, download a [sample archive](https://github.com/supervisely-ecosystem/multiview-object-matching/releases/download/v0.9.4/multiview_example.tar).

### Option 5: Create multi-view project with Python

To create multi-view images project with Supervisely's SDK, follow the [tutorial](https://developer.supervisely.com/getting-started/python-sdk-tutorials/images/multiview-images) in our developer portal.

## Acknowledgements

This application uses [LightGlue](https://github.com/cvg/LightGlue)![GitHub Org's stars](https://img.shields.io/github/stars/cvg/LightGlue?style=social) (ICCV 2023). A lightweight neural network designed for image feature matching. It is used in conjunction with a feature descriptor, specifically, **SuperPoint**. LightGlue and SuperPoint form a powerful pipeline for various computer vision tasks such as image matching and localization. It achieves state-of-the-art performance on several benchmarks while being faster than previous methods.
