
"""Interactive GUI helpers for selecting calibration points.

This module provides Tk-based interfaces used during manual camera-LiDAR
extrinsic calibration.
"""

import cv2

import numpy as np

import tkinter as tk
from tkinter import ttk
from numpy import linalg as la

import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class SelectPointsInterface:
  """Interactive GUI for selecting LiDAR points in the current view."""

  def __init__(self, laser, laser_points):
    """Initialize the LiDAR point-selection interface.

    Args:
        laser: LiDAR scan points as a NumPy array.
        laser_points: List that accumulates selected point groups.

    Returns:
        None.
    """
    self.root = tk.Tk()
    self.root.title("Camera 2D LiDAR Calibration Menu - LiDAR 2D Point Selection")
    self.root.geometry("700x700")
    self.menubar = tk.Menu(self.root)        
    self.app = tk.Frame(self.root)

    # GUI
    window_text = tk.Label(self.root, text=f"Select 2D LiDAR points corresponding to the perpendicular surface of the checkerboard\nby using the Zoom feature followed by clicking 'Select Points'.\nOnce finished, click 'Done'.")
    window_text.pack(padx=5, pady=5)
    
    self.button_frame = ttk.Frame(self.root)    
    self.button_frame.pack(side="top", pady=(20,0))

    self.select_points_button = ttk.Button(self.button_frame, text="Select Points")
    self.select_points_button.pack(side="left", padx=25,pady=(0,10), ipadx=20, ipady=20)
    self.select_points_button.bind("<ButtonPress>", self.select_points)

    self.done_button = ttk.Button(self.button_frame, text="Done")
    self.done_button.pack(side="left", padx=25,pady=(0,10), ipadx=20, ipady=20)
    self.done_button.bind("<ButtonPress>", self.done_callback)

    # Data storage
    self.laser = laser.copy()
    self.laser_2d = None
    self.laser_points = laser_points.copy()
    self.confirmed = False
    
    self.ax_lidar_points = None
    self.ax_selected_lidar_points = None
    self.selected_points_indices = None

    self.add_figure()

    self.reset_and_add_2d_lidar_points()

  def reset_and_add_2d_lidar_points(self):
    """Reset the plot and redraw all LiDAR points.

    Returns:
        None.
    """
    if self.ax_lidar_points:
      self.ax_lidar_points.remove()
    self.ax_lidar_points = None

    if self.ax_selected_lidar_points:
      self.ax_selected_lidar_points.remove()
    self.ax_selected_lidar_points = None
    self.selected_points_indices = None

    if self.laser is not None:
      assert isinstance(self.laser, np.ndarray), "Error: self.laser must be a numpy array."
      self.laser_2d = np.array([[point[0], point[1]] for point in self.laser])
      self.ax_lidar_points = self.ax.scatter(self.laser_2d[:,0], self.laser_2d[:,1], c='blue')

    self.figure.canvas.draw()
    self.figure.canvas.flush_events()

  def select_points(self, event):
    """Select LiDAR points inside the current zoom bounds.

    Args:
      event: Matplotlib button press event.

    Returns:
      None.
    """
    if self.ax_selected_lidar_points:
      self.ax_selected_lidar_points.remove()
    self.ax_selected_lidar_points = None
    self.selected_points_indices = None

    points_xy = self.laser_2d.copy()

    selected_x_indices = np.logical_and(points_xy[:,0]>= self.xlims[0], points_xy[:,0] <= self.xlims[1])
    selected_y_indices = np.logical_and(points_xy[:,1] >= self.ylims[0], points_xy[:,1] <= self.ylims[1])

    self.selected_points_indices = list(np.logical_and(selected_x_indices, selected_y_indices))
    points_xy_selected = points_xy[np.array(self.selected_points_indices)]

    self.ax_selected_lidar_points = self.ax.scatter(points_xy_selected[:,0], points_xy_selected[:,1], c='red')

    self.figure.canvas.draw()
    self.figure.canvas.flush_events()
  
  def on_xlims_change(self, event_ax):
    """Update the stored x-axis limits.

    Args:
        event_ax: Matplotlib axis instance.

    Returns:
        None.
    """
    self.xlims = event_ax.get_xlim()

  def on_ylims_change(self, event_ax):
    """Update the stored y-axis limits.

    Args:
        event_ax: Matplotlib axis instance.

    Returns:
        None.
    """
    self.ylims = event_ax.get_ylim()

  def add_figure(self):
    """Create the Matplotlib figure used by the interface.

    Returns:
        None.
    """
    self.figure = plt.Figure(figsize=(7, 5), dpi=100)
    self.ax = self.figure.add_subplot(111)
    self.chart_type = FigureCanvasTkAgg(self.figure, self.root)
    self.navigation_tool_bar = tkagg.NavigationToolbar2Tk(self.chart_type, self.root)
    self.navigation_tool_bar.zoom()
    self.chart_type.get_tk_widget().pack()
    self.ax.set_title('2D LiDAR Scanned Points')
    self.ax.callbacks.connect('xlim_changed', self.on_xlims_change)
    self.ax.callbacks.connect('ylim_changed', self.on_ylims_change)
    self.ax.set_aspect('equal')
    self.ax.set_xlim([-3.0, 3.0])
    self.ax.set_ylim([-3.0, 3.0])
    self.ax.grid(True)
    self.ax.set_xlabel("X")
    self.ax.set_ylabel("Y")
    pass

  def done_callback(self, event):
    """Confirm the selection and close the window.

    Args:
        event: Matplotlib button press event.

    Returns:
        None.
    """
    if self.selected_points_indices:
      # Extract the selected points and insert them into the output list
      selected_pc2_points = self.laser_2d[self.selected_points_indices]
      self.laser_points.append(selected_pc2_points)
      self.confirmed = True
      self.root.after(0, self.root.destroy)

  def run(self) -> np.ndarray:
    """Run the GUI and return the selected point groups.

    Returns:
        List of selected 2D LiDAR point arrays.
    """
    self.app.mainloop()
    assert self.confirmed, "Error: You must select 2D LiDAR points to proceed."
    return self.laser_points
    
class ImageVisInterface:
  """Interactive GUI for verifying checkerboard pose and extracting points."""

  def __init__(self, rotation_rod, translation, camera_image, camera_points):
    """Initialize the camera verification interface.

    Args:
        rotation_rod: Checkerboard rotation vector in Rodrigues form.
        translation: Checkerboard translation vector.
        camera_image: Camera image used for verification.
        camera_points: List that accumulates extracted camera points.

    Returns:
        None.
    """
    self.root = tk.Tk()
    self.root.title("Camera 2D LiDAR Calibration - Camera View")
    self.root.geometry("700x700")
    self.menubar = tk.Menu(self.root)        
    self.app = tk.Frame(self.root)

    window_text = tk.Label(self.root,
                        text=f"Verify whether this extracted checkerboard is reasonable.\nLook for alignment between extracted corners and the checkerboard, as well as the pose of the board axis.\nOnce verified, click 'Done' to extract a horizonal line in 3D.")
    window_text.pack(padx=5, pady=5)

    self.button_frame = ttk.Frame(self.root)    
    self.button_frame.pack(side="top", pady=(20,0))

    self.select_left_edge_button = ttk.Button(self.button_frame, text="That looks wrong!")
    self.select_left_edge_button.pack(side="left", padx=25, pady=(0,10), ipadx=20, ipady=20)
    self.select_left_edge_button.bind("<ButtonPress>", self.cancel_callback)

    self.done_button = ttk.Button(self.button_frame, text="Done")
    self.done_button.pack(side="left", padx=25, pady=(0,10), ipadx=20, ipady=20)
    self.done_button.bind("<ButtonPress>", self.done_callback)

    # Line extraction data storage
    self.rotation_rod = rotation_rod.copy() # rotation of the camera pose in broad frame - Rodrigues form
    self.translation = translation.copy() # translation of the camera pose in broad frame
    self.camera_points = camera_points.copy() # extracted points, accumulated in every call of this class

    self.add_figure()

    # Flags used to say that the checkerboard detection is not good enough
    self.verified = False
    self.extracted = False

    img = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
    self.ax.imshow(img)


  def add_figure(self) -> None:
    """Create the Matplotlib figure used by the interface.

    Returns:
        None.
    """
    self.figure = plt.Figure(figsize=(7, 5), dpi=100)
    self.ax = self.figure.add_subplot(111)
    self.chart_type = FigureCanvasTkAgg(self.figure, self.root)
    self.navigation_tool_bar = tkagg.NavigationToolbar2Tk(self.chart_type, self.root)
    self.navigation_tool_bar.zoom()
    self.chart_type.get_tk_widget().pack()

  def on_xlims_change(self, event_ax) -> None:
    """Update the stored x-axis limits.

    Args:
      event_ax: Matplotlib axis instance.

    Returns:
      None.
    """
    self.xlims = event_ax.get_xlim()

  def on_ylims_change(self, event_ax) -> None:
    """Update the stored y-axis limits.

    Args:
      event_ax: Matplotlib axis instance.

    Returns:
      None.
    """
    self.ylims = event_ax.get_ylim()

  def done_callback(self, event) -> None:
    """Confirm the image and extract camera-frame line points.

    Args:
      event: Matplotlib button press event.

    Returns:
      None.
    """
    rot_rod_zn90 = np.array([[0], [0], [-np.pi/2]])
    rot_zn90, _ = cv2.Rodrigues(rot_rod_zn90)
    rot_rod_xn90 = np.array([[-np.pi/2], [0], [0]])
    rot_xn90, _ = cv2.Rodrigues(rot_rod_xn90)
    rot_cam_to_robot = rot_zn90 @ rot_xn90 # the rotation that transforms a point in the cv convention to that in the robot convention
    tf_cam_to_robot = np.eye(4)
    tf_cam_to_robot[0:3, 0:3] = rot_cam_to_robot
    # print(tf_cam_to_robot)

    # Compute the tf from the checkerboard to camera - used to transform a point in checkerboard frame to camera frame
    # Also known as the pose of the checkerboard in camera frame
    rotation, _ = cv2.Rodrigues(self.rotation_rod)
    tf_board_to_cam = np.eye(4)
    tf_board_to_cam[0:3, 0:3] = rotation
    tf_board_to_cam[0:3, 3:4] = self.translation
    # print(tf_board_to_cam)

    # The tf from the board frame in robotic convention, to the robot frame (camera frame) in robotic convention
    # The pose of the board frame in robotic convention, in the robot frame (camera frame), in robotic convention
    # Used to transform a point in board frame into camera frame
    tf_robot_board_to_robot = tf_cam_to_robot @ tf_board_to_cam
    # print(tf_robot_board_to_robot)

    # Let's extract/compute a line that goes from the board's origin
    # along the positive direction of the y axis for 30 cm, and negative for 10 cm
    # spacing 0.5 cm
    # This might differ with your board layout, but because we are doing line fitting, so wall length is agnostic
    # It will help with ICP though - more points to fit
    line_end = 0.3
    line_start = -0.1
    line_spacing = 0.005
    board_origin = (tf_robot_board_to_robot @ (np.array([0,0,0,1]).reshape(4,1)))[:3,:]
    #board_y_direction changed from [0,1,0,1] to [1,0,0,1] to extract camera points along the board x-axis
    board_y_direction = (tf_robot_board_to_robot @ (np.array([1,0,0,1]).reshape(4,1)))[:3,:] - board_origin
    board_y_direction = board_y_direction / la.norm(board_y_direction) # normalise the direction just in case - it should be a unit vector to behind with
    line_base = np.linspace(line_start, line_end, int((line_end - line_start)/line_spacing)+1)
    line_points = board_origin.T + np.outer(line_base, board_y_direction)
    line_points_2d = line_points[:, :2]
    self.camera_points.append(line_points_2d)

    self.verified = True
    self.extracted = True
    self.root.after(0, self.root.destroy)

  def cancel_callback(self, event) -> None:
    """Reject the current image and close the window.

    Args:
        event: Matplotlib button press event.

    Returns:
        None.
    """
    self.verified = True
    self.extracted = False
    self.root.after(0, self.root.destroy)


  def run(self) -> tuple[bool, np.ndarray]:
    """Run the GUI and return the extraction result.

    Returns:
        Tuple of (extracted, camera_points).
    """
    self.app.mainloop()
    assert self.verified, "Error: You must verify the quality of the image."
    return self.extracted, self.camera_points
