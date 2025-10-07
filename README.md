# SPILL: Size, Pose, and Internal Liquid Level Estimation of Transparent Glassware for Robotic Bartending

Robotic perception of transparent objects presents unique challenges due to their refractive properties, lack of texture, and limitations of conventional RGB-D sensors in capturing reliable depth information. These challenges significantly hinder robotic manipulation capabilities in real-world settings such as household assistance, hospitality, and healthcare. 

To address these issues, we propose SPILL: A lightweight perception pipeline for \underline{S}ize, \underline{P}ose, and \underline{I}nternal \underline{L}iquid \underline{L}evel estimation of unknown transparent glassware using a single view. SPILL combines object detection with semantic keypoint detection, and operates without requiring object-specific 3D models or depth completion. We demonstrate its effectiveness in autonomous robotic pouring tasks. 

Additionally, to enhance the robustness and generalization of keypoint detection to diverse real-world scenarios, we introduce \textit{Glasses-in-the-Wild}, a new dataset that captures a wide variety of glass types in realistic environments. Evaluated on a robot manipulator, SPILL achieves a 93.6\% success rate across 500 autonomous pours with 20 unseen glasses in three diverse real-world scenes. 

We further demonstrate robustness through multiple live public events in real-world, human-centered environments. In one recorded session, the robot autonomously served 62 drinks with a 98.3\% success rate. 
These results demonstrate that task-relevant keypoint detection enables scalable, real-world transparent object interaction, paving the way for practical applications in service and assistive robotics - without spilling a drop. 

Dataset: [ 10.5281/zenodo.17288314](https://doi.org/10.5281/zenodo.17288314) 
