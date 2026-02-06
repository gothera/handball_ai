from fastcore.basics import patch
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import cv2
import numpy as np

class HandballPitch:
    PITCH_LENGTH = 40.0
    PITCH_WIDTH = 20.0
    GOAL_WIDTH = 3.0
    GOAL_AREA_DISTANCE = 6.0
    FREE_THROW_DISTANCE = 9.0
    PENALTY_DISTANCE = 7.0
    GOALKEEPER_DISTANCE = 4.0
    PENALTY_LINE_LENGTH = 1.0
    GOALKEEPER_LINE_LENGTH = 0.15
    FREE_THROW_SIDELINE_INTERSECTION_LENGTH = 2.958
    CENTER_CIRCLE_RADIUS = 2.0

    def __init__(self):
        hw,hl,gw = self.PITCH_WIDTH/2, self.PITCH_LENGTH/2, self.GOAL_WIDTH/2

        self.keypoints = dict(
            corner_left_up=(-hl, hw), corner_left_down=(-hl, -hw),
            corner_right_up=(hl, hw), corner_right_down=(hl, -hw),
            seven_left_up=(-hl + self.PENALTY_DISTANCE, self.PENALTY_LINE_LENGTH/2.0), seven_left_down=(-hl + self.PENALTY_DISTANCE, -self.PENALTY_LINE_LENGTH/2.0),
            seven_right_up=(hl - self.PENALTY_DISTANCE, self.PENALTY_LINE_LENGTH/2.0), seven_right_down=(hl - self.PENALTY_DISTANCE, -self.PENALTY_LINE_LENGTH/2.0),
            four_left_up=(-hl + self.GOALKEEPER_DISTANCE, self.GOALKEEPER_LINE_LENGTH/2.0), four_left_down=(-hl + self.GOALKEEPER_DISTANCE, -self.GOALKEEPER_LINE_LENGTH/2.0),
            four_right_up=(hl - self.GOALKEEPER_DISTANCE, self.GOALKEEPER_LINE_LENGTH/2.0), four_right_down=(hl - self.GOALKEEPER_DISTANCE, -self.GOALKEEPER_LINE_LENGTH/2.0),
            six_goalline_left_up=(-hl, self.GOAL_AREA_DISTANCE + gw), six_goalline_left_down=(-hl, -self.GOAL_AREA_DISTANCE - gw),
            six_goalline_right_up=(hl, self.GOAL_AREA_DISTANCE + gw), six_goalline_right_down=(hl, -self.GOAL_AREA_DISTANCE - gw),
            nine_sideline_left_up=(-hl + self.FREE_THROW_SIDELINE_INTERSECTION_LENGTH, hw), nine_sideline_left_down=(-hl + self.FREE_THROW_SIDELINE_INTERSECTION_LENGTH, -hw),
            nine_sideline_right_up=(hl - self.FREE_THROW_SIDELINE_INTERSECTION_LENGTH, hw), nine_sideline_right_down=(hl - self.FREE_THROW_SIDELINE_INTERSECTION_LENGTH, -hw),
            middle_sideline_up=(0.0, hw), middle_sideline_down=(0.0, -hw),
            goal_post_left_up=(-hl, gw), goal_post_left_down=(-hl, -gw),
            goal_post_right_up=(hl, gw), goal_post_right_down=(hl, -gw),
            center_circle_up=(0, self.CENTER_CIRCLE_RADIUS), center_circle_down=(0, -self.CENTER_CIRCLE_RADIUS)
            # seven_sideline_right_up=(hl - self.PENALTY_DISTANCE, hw),
            # nine_parallel_right_up=(hl - self.FREE_THROW_DISTANCE, gw), nine_parallel_right_down=(hl - self.FREE_THROW_DISTANCE, -gw),
            # six_goalline_parallel_right_up=(hl - self.GOAL_AREA_DISTANCE, gw), six_goalline_parallel_right_down=(hl - self.GOAL_AREA_DISTANCE, -gw),
            # four_sideline_intersection_right_up=(hl - self.GOALKEEPER_DISTANCE, hw), four_sideline_intersection_right_down=(hl - self.GOALKEEPER_DISTANCE, -hw)
        )

        self.keypoint_ids = {k: i for i,k in enumerate(self.keypoints.keys(), 1)}

@patch
def draw_pitch(self:HandballPitch, ax=None, figsize=(12,8), save_path=None):
    "Draw handball pitch with all lines and keypoints"
    if ax is None: fig,ax = plt.subplots(figsize=figsize)
    hw,hl,gw = self.PITCH_WIDTH/2, self.PITCH_LENGTH/2, self.GOAL_WIDTH/2

    ax.plot([-hl,-hl,hl,hl,-hl], [-hw,hw,hw,-hw,-hw], 'k-', linewidth=2)
    ax.plot([0,0], [-hw,hw], 'k-', linewidth=2)

    ax.plot([-hl+self.PENALTY_DISTANCE,-hl+self.PENALTY_DISTANCE], [-0.5,0.5], 'k-', linewidth=2)
    ax.plot([hl-self.PENALTY_DISTANCE,hl-self.PENALTY_DISTANCE], [-0.5,0.5], 'k-', linewidth=2)

    ax.plot([-hl+self.GOALKEEPER_DISTANCE,-hl+self.GOALKEEPER_DISTANCE], [-0.075,0.075], 'k-', linewidth=2)
    ax.plot([hl-self.GOALKEEPER_DISTANCE,hl-self.GOALKEEPER_DISTANCE], [-0.075,0.075], 'k-', linewidth=2)

    ax.plot([-hl+self.GOAL_AREA_DISTANCE,-hl+self.GOAL_AREA_DISTANCE], [gw,-gw], 'k-', linewidth=2)
    ax.plot([hl-self.GOAL_AREA_DISTANCE,hl-self.GOAL_AREA_DISTANCE], [gw,-gw], 'k-', linewidth=2)

    ax.plot([-hl+self.FREE_THROW_DISTANCE,-hl+self.FREE_THROW_DISTANCE], [gw,-gw], 'k--', linewidth=2)
    ax.plot([hl-self.FREE_THROW_DISTANCE,hl-self.FREE_THROW_DISTANCE], [gw,-gw], 'k--', linewidth=2)

    arc_left_up = Arc((-hl, gw), 2*self.GOAL_AREA_DISTANCE, 2*self.GOAL_AREA_DISTANCE, angle=0, theta1=0, theta2=90, color='k', linewidth=2)
    arc_left_down = Arc((-hl, -gw), 2*self.GOAL_AREA_DISTANCE, 2*self.GOAL_AREA_DISTANCE, angle=0, theta1=270, theta2=360, color='k', linewidth=2)
    arc_right_up = Arc((hl, gw), 2*self.GOAL_AREA_DISTANCE, 2*self.GOAL_AREA_DISTANCE, angle=0, theta1=90, theta2=180, color='k', linewidth=2)
    arc_right_down = Arc((hl, -gw), 2*self.GOAL_AREA_DISTANCE, 2*self.GOAL_AREA_DISTANCE, angle=0, theta1=180, theta2=270, color='k', linewidth=2)
    ax.add_patch(arc_left_up); ax.add_patch(arc_left_down); ax.add_patch(arc_right_up); ax.add_patch(arc_right_down)

    nine_dist = self.GOAL_AREA_DISTANCE + 3
    arc_nine_left_up = Arc((-hl, gw), 2*nine_dist, 2*nine_dist, angle=0, theta1=0, theta2=90, color='k', linewidth=2, linestyle='--')
    arc_nine_left_down = Arc((-hl, -gw), 2*nine_dist, 2*nine_dist, angle=0, theta1=270, theta2=360, color='k', linewidth=2, linestyle='--')
    arc_nine_right_up = Arc((hl, gw), 2*nine_dist, 2*nine_dist, angle=0, theta1=90, theta2=180, color='k', linewidth=2, linestyle='--')
    arc_nine_right_down = Arc((hl, -gw), 2*nine_dist, 2*nine_dist, angle=0, theta1=180, theta2=270, color='k', linewidth=2, linestyle='--')
    ax.add_patch(arc_nine_left_up); ax.add_patch(arc_nine_left_down); ax.add_patch(arc_nine_right_up); ax.add_patch(arc_nine_right_down)

    for name,(x,y) in self.keypoints.items():
        keypoint_id = str(self.keypoint_ids[name])
        ax.plot(x, y, 'ro', markersize=4)
        if keypoint_id == "10" or keypoint_id == "12":
            ax.text(x-0.5, y-0.5, keypoint_id, fontsize=8, ha='right', va='center') 
            continue
        ax.text(x-0.5, y+0.5, keypoint_id, fontsize=8, ha='right', va='center')

    ax.set_aspect('equal'); ax.set_xlim(-hl, hl); ax.set_ylim(-hw, hw); ax.grid(True, alpha=0.3); ax.set_axis_off();

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)

    return ax

@patch
def find_homography(self:HandballPitch, img_pts):
    "Find homography matrix from image points to pitch coordinates"
    pitch_pts = np.array([self.keypoints[k] for k in img_pts.keys()], dtype=np.float32)
    img_pts_arr = np.array(list(img_pts.values()), dtype=np.float32)
    H,_ = cv2.findHomography(img_pts_arr, pitch_pts)
    return H

@patch
def show_warped_pitch_on_image(self:HandballPitch, image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig,ax = plt.subplots(figsize=(14,10))
    ax.imshow(img)

    pitch_corners = np.array([[-20,-10], [20,-10], [20,10], [-20,10], [-20,-10]], dtype=np.float32)
    img_corners = cv2.perspectiveTransform(pitch_corners.reshape(-1,1,2), np.linalg.inv(H)).reshape(-1,2)
    ax.plot(img_corners[:,0], img_corners[:,1], 'r-', linewidth=2)

    for name,pt in self.keypoints.items():
        img_pt = cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), np.linalg.inv(H))[0,0]
        ax.plot(img_pt[0], img_pt[1], 'ro', markersize=4)

    ax.set_axis_off()     

@patch
def warp_template_to_image(self:HandballPitch, template_path, img_path, img_pts, ax=None, figsize=(12,8)):
    "Warp pitch template image onto target image using homography"
    template = plt.imread(template_path)
    img = plt.imread(img_path)
    H = self.find_homography(img_pts)
    H_inv = np.linalg.inv(H)

    th,tw = template.shape[:2]
    hw,hl = self.PITCH_WIDTH/2, self.PITCH_LENGTH/2
    scale_x,scale_y = tw/(2*hl), th/(2*hw)

    T_template = np.array([[scale_x,0,tw/2],[0,scale_y,th/2],[0,0,1]])
    H_combined = H_inv @ np.linalg.inv(T_template)

    warped = cv2.warpPerspective(template, H_combined, (img.shape[1],img.shape[0]))

    if ax is None: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.imshow(warped, alpha=0.5)
    ax.set_axis_off()
    return ax       
