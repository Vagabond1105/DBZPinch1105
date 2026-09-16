"""
ui_ux.py

UI/UX elements: keyboard-driven camera control, the top-left runtime data
display, buttons, parameter sliders, and the init-only tutorial box.

Architecture notes (see chat for full reasoning):
- CameraController is logic-only. It does NOT render or own a camera
  object - it tracks radius/theta/phi from keyboard input, and main.py
  is expected to push those values into VisPy's TurntableCamera
  (azimuth/elevation/distance) each frame. This reconciles ui_ux.py.md's
  keyboard-driven camera spec with visualisation.py.md's explicit
  instruction to rely on VisPy's own built-in camera.
- Buttons and sliders take plain mouse_pos / is_clicked / keys_held
  arguments rather than binding to PyQt5 or VisPy's event systems
  directly - which backend actually captures raw input is a main.py
  wiring decision, not this file's.
- All rendering (sliders, buttons, text, tutorial box) uses vispy scene
  visuals, matching the vispy.scene.widgets.Slider approach already used
  in visualisation.py's reference, rather than mixing in a second,
  separate PyQt5-widget UI layer.
"""

import numpy as np

from vispy.scene import visuals

from constants import cam_radius_min, cam_radius_max, cam_radius_default


# =====================================================
# camera controller
# =====================================================

class CameraController:
    """
    Tracks logical camera state (radius, theta, phi) driven by keyboard
    input. Does not touch VisPy's camera object directly - see module
    docstring.
    """

    def __init__(self):
        self.theta_def = 0.0
        self.phi_def = np.pi / 4.0
        self.r_def = cam_radius_default

        self.theta = self.theta_def
        self.phi = self.phi_def
        self.radius = self.r_def

        self.theta_range = (-np.pi, np.pi)
        self.phi_range = (-np.pi / 2.0, np.pi / 2.0)
        self.radius_range = (cam_radius_min, cam_radius_max)

        self.step_angle = np.deg2rad(1.0)
        self.step_radius = 0.02 * (cam_radius_max - cam_radius_min)

    def detect_input(self, keys_held):
        """
        keys_held: a set/iterable of currently-held key-name strings
        (e.g. {"Left", "W", "+"}), populated by whichever windowing layer
        main.py wires up to capture key state.

        Returns one of "left"/"right"/"up"/"down"/"zoom_in"/"zoom_out",
        or False if nothing camera-relevant is currently held. Only one
        action is returned per call, checked in a fixed priority order -
        good enough for a keyboard-orbit camera, not meant to support
        diagonal combos.
        """
        if "Left" in keys_held or "A" in keys_held or "a" in keys_held:
            return "left"
        if "Right" in keys_held or "D" in keys_held or "d" in keys_held:
            return "right"
        if "Up" in keys_held or "W" in keys_held or "w" in keys_held:
            return "up"
        if "Down" in keys_held or "S" in keys_held or "s" in keys_held:
            return "down"
        if "+" in keys_held or "=" in keys_held:
            return "zoom_in"
        if "-" in keys_held or "_" in keys_held:
            return "zoom_out"
        return False

    def change_camera(self, action):
        """
        Applies one step of the given action, then clamps every axis to
        its range. NOTE: theta is clamped (snapped), not wrapped, per the
        literal spec ("if at boundary or past it, it will snap to that
        boundary") - worth a second look later, since a yaw angle is
        normally periodic and clamping means the camera can't swing past
        +/-180 degrees to keep turning the same direction; it just stops.
        """
        if not action:
            return

        if action == "left":
            self.theta -= self.step_angle
        elif action == "right":
            self.theta += self.step_angle
        elif action == "up":
            self.phi += self.step_angle
        elif action == "down":
            self.phi -= self.step_angle
        elif action == "zoom_in":
            self.radius -= self.step_radius
        elif action == "zoom_out":
            self.radius += self.step_radius

        theta_min, theta_max = self.theta_range
        self.theta = min(max(self.theta, theta_min), theta_max)

        phi_min, phi_max = self.phi_range
        self.phi = min(max(self.phi, phi_min), phi_max)

        r_min, r_max = self.radius_range
        self.radius = min(max(self.radius, r_min), r_max)


# =====================================================
# runtime data display (top-left, while running)
# =====================================================

def compute_potential_energy(sim_state):
    """
    Gravitational potential energy, computed live rather than tracked as
    a history array. "potential_energy" is listed in the UI spec's
    runtime data block, but nothing in simstate.py/solvers.py tracks it -
    the diagnostics export graphs don't ask for it either, so there was
    no history list to reuse. For uniform gravity g in -y (see
    SimState.gravity_vector), potential energy density is rho * g * y;
    summed the same unweighted way the other energy totals already are
    in solvers.py (no explicit cell-volume factor - consistent with, not
    a departure from, the existing convention).
    """
    g = sim_state.rt_params.grav.value_rn
    pe_density = sim_state.rho * g * sim_state.Y
    return float(np.sum(pe_density))


class RuntimeDataDisplay:
    """
    Manages the stack of labeled metrics shown in the top-left corner
    while the sim is running: time elapsed, core averages, confinement/
    Lawson proxies, and the four energy totals (including potential).
    """

    def __init__(self, parent, start_position=(20, 20), line_spacing=20):
        self.parent = parent
        self.start_position = start_position
        self.line_spacing = line_spacing
        self.labels = {}
        self.order = []

    def _set_value(self, label, value):
        text_str = f"{label}: {value}"

        if label not in self.labels:
            index = len(self.order)
            x, y = self.start_position
            pos = (x, y + index * self.line_spacing)
            self.labels[label] = visuals.Text(
                text_str,
                pos=pos,
                color="white",
                font_size=10,
                anchor_x="left",
                anchor_y="top",
                parent=self.parent,
            )
            self.order.append(label)
        else:
            self.labels[label].text = text_str

    def update_from_simstate(self, sim_state):
        core = sim_state.core_mask
        p = sim_state.pressure()
        T = sim_state.temperature()

        def last_or_zero(history):
            return history[-1] if history else 0.0

        metrics = {
            "Time Elapsed": round(sim_state.t, 4),
            "Core Temperature": round(float(np.mean(T[core])), 4),
            "Core Pressure": round(float(np.mean(p[core])), 4),
            "Core Density": round(float(np.mean(sim_state.rho[core])), 4),
            "Lawson Proxy": round(last_or_zero(sim_state.lawson_evolution), 4),
            "Confinement Proxy": round(last_or_zero(sim_state.confinement_evolution), 4),
            "Kinetic Energy": round(last_or_zero(sim_state.kinetic_energy_history), 4),
            "Potential Energy": round(compute_potential_energy(sim_state), 4),
            "Magnetic Energy": round(last_or_zero(sim_state.magnetic_energy_history), 4),
            "Total Energy": round(last_or_zero(sim_state.total_energy_history), 4),
        }

        for label, value in metrics.items():
            self._set_value(label, value)

    def clear(self):
        for text_visual in self.labels.values():
            text_visual.parent = None
        self.labels = {}
        self.order = []


# =====================================================
# buttons
# =====================================================

class Button:
    """
    A single clickable button: START/PAUSE/RESUME, RESTART, RESTART
    DEFAULT, and EXPORT DATA AND DIAGNOSTICS are all instances of this,
    differing only in text/colour/position/size per the UI spec.
    """

    def __init__(self, text, background_colour, position, length=100, height=50, parent=None):
        self.text = text
        self.background_colour = background_colour
        self.position = position
        self.length = length
        self.height = height
        self.display = False

        self._hover_colour = self._shift(background_colour, 0.15)
        self._click_colour = self._shift(background_colour, -0.15)
        self._current_colour = background_colour
        self._was_down = False

        self.parent = parent
        self._rect_visual = None
        self._text_visual = None

    @staticmethod
    def _shift(colour, amount):
        r, g, b = colour
        clamp = lambda c: min(max(c, 0.0), 1.0)
        return (clamp(r + amount), clamp(g + amount), clamp(b + amount))

    def contains(self, mouse_pos):
        mx, my = mouse_pos
        x, y = self.position
        return x <= mx <= x + self.length and y <= my <= y + self.height

    def detect_interaction(self, mouse_pos, is_clicked):
        """
        Call once per frame with current mouse position and click state.
        Returns True exactly on the FRAME A CLICK STARTS inside this
        button's bounds - not on every frame the mouse happens to stay
        down while hovering (that was a real bug found during main.py
        wiring: at 60Hz, holding the mouse down over a button used to
        fire the click handler dozens of times per second instead of
        once). Also updates the button's colour state for render() to
        pick up - slightly lighter on hover, slightly darker on click.
        """
        if not self.display:
            self._current_colour = self.background_colour
            self._was_down = False
            return False

        hovering = self.contains(mouse_pos)
        clicked_this_frame = is_clicked and not self._was_down and hovering
        self._was_down = is_clicked

        if clicked_this_frame:
            self._current_colour = self._click_colour
            return True

        if hovering:
            self._current_colour = self._click_colour if is_clicked else self._hover_colour
        else:
            self._current_colour = self.background_colour

        return False

    def render(self):
        if not self.display:
            if self._rect_visual is not None:
                self._rect_visual.parent = None
                self._text_visual.parent = None
            return

        x, y = self.position
        center = (x + self.length / 2.0, y + self.height / 2.0)

        if self._rect_visual is None:
            self._rect_visual = visuals.Rectangle(
                center=center,
                width=self.length,
                height=self.height,
                color=self._current_colour,
                parent=self.parent,
            )
            self._text_visual = visuals.Text(
                self.text,
                pos=center,
                color="white",
                font_size=10,
                parent=self.parent,
            )
        else:
            self._rect_visual.parent = self.parent
            self._text_visual.parent = self.parent
            self._rect_visual.color = self._current_colour
            self._text_visual.text = self.text


# =====================================================
# parameter sliders
# =====================================================

class ParameterSlider:
    """
    Shared drag-to-set slider behaviour and rendering. Styled differently
    by colour_type per the UI readme: 0 (init) = black background, white
    outline; 1 (rt) = white background, black outline.

    DEVIATION FROM LITERAL SPEC: ui_ux.py.md says Init_slider/RT_slider
    take separate max/min/default arguments. This takes the actual
    Parameter instance (from parameters.py) instead, since the slider has
    to write back into it anyway via update_value() - passing three loose
    numbers that must always match that object risked them drifting out
    of sync for no benefit.
    """

    LENGTH = 200
    HEIGHT = 40

    def __init__(self, parameter, position, param_type, colour_type, name):
        self.parameter = parameter
        self.name = name
        self.max = parameter.max_value
        self.min = parameter.min_value
        self.default = parameter.default_value
        self.value = parameter.value_rn

        self.position = position
        self.length = self.LENGTH
        self.height = self.HEIGHT
        self.show_bool = True
        self.colour_type = colour_type
        self.param_type = param_type

        self._dragging = False

        self.parent = None
        self._track_visual = None
        self._knob_visual = None
        self._label_visual = None

    def _value_to_knob_x(self):
        frac = (self.value - self.min) / (self.max - self.min)
        x, _ = self.position
        return x + frac * self.length

    def _knob_x_to_value(self, knob_x):
        x, _ = self.position
        frac = (knob_x - x) / self.length
        frac = min(max(frac, 0.0), 1.0)
        return self.min + frac * (self.max - self.min)

    def interact_mouse(self, mouse_pos, is_mouse_down):
        """
        Call once per frame. Starts a drag if the mouse goes down on the
        knob, and while dragging maps mouse x to a new value, applied
        through parameter.update_value() (so range enforcement stays in
        one place - parameters.py - rather than duplicated here).
        """
        if not self.show_bool:
            return

        mx, my = mouse_pos
        knob_x = self._value_to_knob_x()
        _, y = self.position

        near_knob = (
            abs(mx - knob_x) <= self.height / 2.0
            and abs(my - (y + self.height / 2.0)) <= self.height / 2.0
        )

        if is_mouse_down and (self._dragging or near_knob):
            self._dragging = True
            new_value = self._knob_x_to_value(mx)
            self.parameter.update_value(new_value)
            self.value = self.parameter.value_rn
        else:
            self._dragging = False

    def render_slider(self):
        if not self.show_bool:
            if self._track_visual is not None:
                self._track_visual.parent = None
                self._knob_visual.parent = None
                self._label_visual.parent = None
            return

        if self.colour_type == 0:
            bg_colour, outline_colour = "black", "white"
        else:
            bg_colour, outline_colour = "white", "black"

        x, y = self.position
        track_center = (x + self.length / 2.0, y + self.height / 2.0)
        knob_x = self._value_to_knob_x()
        knob_center = (knob_x, y + self.height / 2.0)
        label_pos = (x, y - 14)
        label_text = f"{self.name}: {round(self.value, 3)}"

        if self._track_visual is None:
            self._track_visual = visuals.Rectangle(
                center=track_center,
                width=self.length,
                height=self.height,
                color=bg_colour,
                border_color=outline_colour,
                border_width=2,
                parent=self.parent,
            )
            self._knob_visual = visuals.Ellipse(
                center=knob_center,
                radius=self.height / 2.0,
                color=outline_colour,
                parent=self.parent,
            )
            self._label_visual = visuals.Text(
                label_text,
                pos=label_pos,
                color="white",
                font_size=9,
                anchor_x="left",
                anchor_y="bottom",
                parent=self.parent,
            )
        else:
            self._track_visual.parent = self.parent
            self._knob_visual.parent = self.parent
            self._label_visual.parent = self.parent
            self._knob_visual.center = knob_center
            self._label_visual.text = label_text


class InitSlider(ParameterSlider):
    def __init__(self, parameter, position, name):
        super().__init__(parameter, position, param_type="init", colour_type=0, name=name)


class RTSlider(ParameterSlider):
    def __init__(self, parameter, position, name):
        super().__init__(parameter, position, param_type="rt", colour_type=1, name=name)


# =====================================================
# tutorial box (init only)
# =====================================================

class TutorialBox:
    """
    Static 400x400 text panel, bottom-right, visible only while
    sim_status == "init". Purely informational - no effect on the sim.
    """

    WIDTH = 400
    HEIGHT = 400

    DEFAULT_TEXT = (
        "DBZPinch1105 - Z-Pinch MHD Simulator\n\n"
        "Controls:\n"
        "  Arrow keys / WASD - orbit camera\n"
        "  +/- - zoom camera in/out\n"
        "  Drag sliders - adjust parameters\n\n"
        "Adjust the Init sliders (left) before starting, then press "
        "START.\nRuntime sliders (right) can be changed at any time."
    )

    def __init__(self, position, parent=None):
        self.position = position
        self.width = self.WIDTH
        self.height = self.HEIGHT
        self.text = self.DEFAULT_TEXT
        self.parent = parent
        self._text_visual = None

    def render(self, visible):
        if not visible:
            if self._text_visual is not None:
                self._text_visual.parent = None
            return

        if self._text_visual is None:
            self._text_visual = visuals.Text(
                self.text,
                pos=self.position,
                color="white",
                font_size=9,
                anchor_x="left",
                anchor_y="top",
                parent=self.parent,
            )
        else:
            self._text_visual.parent = self.parent
            self._text_visual.text = self.text