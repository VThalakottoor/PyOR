"""
PyOR Python On Resonance

Author: Vineeth Francis Thalakottoor Jose Chacko

Email: vineethfrancis.physics@gmail.com

Description:
    This file contains the classes `RFCircuit`, `Element` and `TransmissionLine`.
"""

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

__all__ = ["Element", "TransmissionLine", "CoaxialLine", "RFCircuit"]


PREFIX = {"": 1, "k": 1e3, "M": 1e6, "G": 1e9,
          "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}
GROUND = "P0"
LIGHT_SPEED = 299792458.0
LENGTH_UNITS = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}

UNIT_SCALE = {"Ω": 1, "kΩ": 1e3, "MΩ": 1e6,
              "pH": 1e-12, "nH": 1e-9, "µH": 1e-6, "mH": 1e-3, "H": 1,
              "fF": 1e-15, "pF": 1e-12, "nF": 1e-9, "µF": 1e-6, "mF": 1e-3, "F": 1}
UNIT_SCALE.update(LENGTH_UNITS)
KIND_UNITS = {"R": ("Ω", "kΩ", "MΩ"),
              "L": ("pH", "nH", "µH", "mH", "H"),
              "C": ("fF", "pF", "nF", "µF", "mF", "F")}


def normalize_node(node):
    node = str(node).strip()
    return GROUND if node.upper() in ("0", "P0") else node


def component_value(e):
    if isinstance(e, TransmissionLine):
        return (f"{e.value/UNIT_SCALE[e.display_unit]:.5g} {e.display_unit}; {e.z0:g} Ω\n"
                f"VF={e.velocity_factor:.4g}; er={e.epsilon_r:.4g}")
    units = KIND_UNITS[e.kind]
    if e.display_unit in units:
        unit = e.display_unit
    else:
        unit = next((u for u in reversed(units) if e.value >= UNIT_SCALE[u]), units[0])
    return f"{e.value/UNIT_SCALE[unit]:.5g} {unit}"


def _draw_coax_symbol(ax, mx, my, ux, uy, half, *, color="black", font_size=10):
    """Rounded coaxial shield, circular input face and separate P0 bond."""
    nx, ny = -uy, ux
    radius = min(.22, half*.28)
    body_half = half*.55

    def draw(local_x, local_y, **options):
        local_x = np.asarray(local_x)
        local_y = np.asarray(local_y)
        ax.plot(mx+ux*local_x+nx*local_y,
                my+uy*local_x+ny*local_y, color=color, **options)

    # Straight shell edges and a rounded far end.
    for side in (-1, 1):
        draw([-body_half, body_half], [side*radius, side*radius], lw=1.6)
    theta = np.linspace(-np.pi/2, np.pi/2, 65)
    draw(body_half+radius*np.cos(theta), radius*np.sin(theta), lw=1.6)
    # The front face is circular, as in the supplied symbol.
    theta = np.linspace(0, 2*np.pi, 129)
    draw(-body_half+radius*np.cos(theta), radius*np.sin(theta), lw=1.6)
    # Exposed signal leads; the conductor inside the shield is hidden.
    draw([-half, -body_half], [0, 0], lw=1.6)
    draw([body_half+radius, half], [0, 0], lw=1.6)
    if abs(ux) >= abs(uy):
        sx, sy = mx, my-radius
        gx, gy = sx, sy-.55
        ax.plot([sx,gx], [sy,gy], color=color, lw=1.4)
    else:
        sx, sy = mx+radius, my
        gx, gy = sx+.60, sy-.45
        ax.plot([sx,gx,gx], [sy,sy,gy], color=color, lw=1.4)
    for offset, width in ((0,.23), (-.09,.16), (-.18,.08)):
        ax.plot([gx-width,gx+width], [gy+offset,gy+offset], color=color, lw=1.4)
    ax.text(gx,gy-.30,GROUND,ha="center",va="top",fontsize=font_size,
            color=color,fontweight="bold")


def parse_value(text):
    match = re.fullmatch(r"\s*(\d+(?:\.\d*)?|\.\d+)(?:[eE]([+-]?\d+))?"
                         r"\s*([kM Gmuµnpf]?)(?:\s*(?:Hz|ohm|Ω|[HFR]))?\s*",
                         str(text).replace(" ", ""))
    if not match:
        raise ValueError(f"Invalid value {text!r}; try 50, 100nH, or 10pF")
    prefix = match[3]
    if prefix == " ":
        prefix = ""
    result = float(match[1]) * 10 ** int(match[2] or 0) * PREFIX[prefix]
    if not np.isfinite(result) or result <= 0:
        raise ValueError("Every component value and frequency must be positive and finite")
    return result


@dataclass(init=False)
class Element:
    name: str
    kind: str
    node_a: str
    node_b: str
    value: float
    display_unit: str = ""

    def __new__(cls, name, kind, node_a, node_b, value, unit="", **line_parameters):
        """Use Element(..., 'TL', ...) for a uniform transmission line/coax."""
        if str(kind).strip().upper() in ("TL", "COAX"):
            return TransmissionLine(name, node_a, node_b, value,
                                    str(unit).strip() or "m", **line_parameters)
        if line_parameters:
            raise TypeError("Extra line parameters apply only to kind TL or COAX")
        return object.__new__(cls)

    def __init__(self, name, kind, node_a, node_b, value, unit=""):
        """Define a component with an explicit unit, e.g. value=0.3045, unit='pF'.

        With no unit, value uses SI (ohms, henries or farads). Internally .value
        always stores SI so a unit is converted exactly once.
        """
        self.name = str(name).strip()
        self.kind = str(kind).strip().upper()
        self.node_a, self.node_b = normalize_node(node_a), normalize_node(node_b)
        if self.kind not in KIND_UNITS:
            raise ValueError("Component kind must be R, L, C, TL or COAX")
        unit = str(unit).strip().replace("μ", "µ")
        unit = {"ohm":"Ω", "Ohm":"Ω", "ohms":"Ω", "uH":"µH", "uF":"µF"}.get(unit,unit)
        if unit and unit not in KIND_UNITS[self.kind]:
            raise ValueError(f"Invalid unit {unit!r} for {self.kind}; use {KIND_UNITS[self.kind]}")
        self.value = float(value) * (UNIT_SCALE[unit] if unit else 1.0)
        self.display_unit = unit
        if not self.name or not self.node_a or not self.node_b or self.node_a == self.node_b:
            raise ValueError("Use a component name and two distinct node names")
        if not np.isfinite(self.value) or self.value <= 0:
            raise ValueError("Component value must be positive and finite")

    @classmethod
    def From_SI(cls, name, kind, node_a, node_b, value, display_unit=""):
        """Build from saved/internal SI data without converting it again."""
        element = cls(name,kind,node_a,node_b,value)
        if display_unit and display_unit not in KIND_UNITS[element.kind]:
            raise ValueError("Invalid display unit in saved component")
        element.display_unit = display_unit
        return element


class TransmissionLine:
    """Uniform TEM line or coax with a common ground/shield reference P0.

    Length uses mm, cm or m. Supply epsilon_r OR velocity_factor; the
    other is derived assuming relative permeability one. Characteristic
    impedance is real and attenuation in dB/m is constant with frequency.
    """
    kind = "TL"

    def __init__(self, name, node_a, node_b, length, unit="m", *, z0=50,
                 epsilon_r=None, velocity_factor=None, attenuation_db_per_m=0):
        self.name = str(name).strip()
        self.node_a, self.node_b = normalize_node(node_a), normalize_node(node_b)
        self.display_unit = str(unit).strip()
        if self.display_unit not in LENGTH_UNITS:
            raise ValueError("Line length unit must be mm, cm or m")
        if epsilon_r is not None and velocity_factor is not None:
            raise ValueError("Supply epsilon_r or velocity_factor, not both")
        if epsilon_r is not None and (not np.isfinite(float(epsilon_r)) or float(epsilon_r) < 1):
            raise ValueError("Relative dielectric constant must be finite and >= 1")
        self.value = float(length)*LENGTH_UNITS[self.display_unit]
        self.z0 = float(z0)
        self.velocity_factor = (float(velocity_factor) if velocity_factor is not None
                                else 1/np.sqrt(float(epsilon_r)) if epsilon_r is not None else 1.0)
        self.attenuation_db_per_m = float(attenuation_db_per_m)
        self.Validate()

    @property
    def length(self):
        """Physical length in metres."""
        return self.value

    @length.setter
    def length(self, value):
        self.value = float(value)

    @property
    def epsilon_r(self):
        return 1/self.velocity_factor**2

    @epsilon_r.setter
    def epsilon_r(self, value):
        value = float(value)
        if not np.isfinite(value) or value < 1:
            raise ValueError("Relative dielectric constant must be finite and >= 1")
        self.velocity_factor = 1/np.sqrt(value)

    @property
    def velocity(self):
        """Propagation velocity in metres per second."""
        return LIGHT_SPEED*self.velocity_factor

    def Validate(self):
        if not self.name or not self.node_a or not self.node_b or self.node_a == self.node_b:
            raise ValueError("Use a line name and two distinct endpoint nodes")
        if (not np.all(np.isfinite([self.value, self.z0, self.velocity_factor,
                                   self.attenuation_db_per_m])) or self.value <= 0 or
            self.z0 <= 0 or not 0 < self.velocity_factor <= 1 or self.attenuation_db_per_m < 0):
            raise ValueError("Line length and Z0 must be positive; 0 < VF <= 1; loss >= 0")

    def Delay(self):
        """One-way propagation delay in seconds."""
        self.Validate()
        return self.length/self.velocity

    def Electrical_Length(self, frequency, unit="MHz"):
        """Electrical length in degrees at the given frequency."""
        scales = {"Hz":1, "kHz":1e3, "MHz":1e6, "GHz":1e9}
        if unit not in scales:
            raise ValueError("Frequency unit must be Hz, kHz, MHz or GHz")
        f = np.asarray(frequency, dtype=float)*scales[unit]
        if np.any(f <= 0) or not np.all(np.isfinite(f)):
            raise ValueError("Frequency must be positive and finite")
        return 360*f*self.Delay()

    def _parameters(self):
        return dict(name=self.name, kind="TL", node_a=self.node_a, node_b=self.node_b,
                    value=self.value/LENGTH_UNITS[self.display_unit], unit=self.display_unit,
                    z0=self.z0, velocity_factor=self.velocity_factor,
                    attenuation_db_per_m=self.attenuation_db_per_m)


CoaxialLine = TransmissionLine


def _load_component(item):
    if item["kind"] == "TL":
        return TransmissionLine(item["name"], item["node_a"], item["node_b"],
            item["value"], item["unit"], z0=item["z0"],
            velocity_factor=item["velocity_factor"],
            attenuation_db_per_m=item.get("attenuation_db_per_m", 0))
    return Element(item["name"], item["kind"], item["node_a"], item["node_b"],
                   item["value"], item["unit"])


def _line_s_parameters(frequency, elements, port_nodes, z0):
    """Modified nodal line equations remain finite at half-wave lengths."""
    f = np.asarray(frequency, dtype=float)
    nodes = list(dict.fromkeys([*port_nodes, *(node for e in elements
                         for node in (e.node_a, e.node_b) if node != GROUND)]))
    index = {node:i for i,node in enumerate(nodes)}
    lines = [e for e in elements if isinstance(e, TransmissionLine)]
    n = len(nodes)
    size = n+2*len(lines)
    matrix = np.zeros((len(f), size, size), dtype=complex)
    incidence = np.zeros((size, len(port_nodes)), dtype=complex)
    for j,node in enumerate(port_nodes):
        incidence[index[node],j] = 1
    matrix += incidence @ incidence.T/z0
    omega = 2*np.pi*f
    for e in elements:
        if isinstance(e, TransmissionLine):
            continue
        branch = ({"R":lambda:np.full(len(f),1/e.value),
                   "C":lambda:1j*omega*e.value,
                   "L":lambda:1/(1j*omega*e.value)})[e.kind]()
        a, b = index.get(e.node_a), index.get(e.node_b)
        if a is not None: matrix[:,a,a] += branch
        if b is not None: matrix[:,b,b] += branch
        if a is not None and b is not None:
            matrix[:,a,b] -= branch
            matrix[:,b,a] -= branch
    for k,line in enumerate(lines):
        a, b = index.get(line.node_a), index.get(line.node_b)
        ia, ib = n+2*k, n+2*k+1
        propagation = (line.attenuation_db_per_m*np.log(10)/20+
                       1j*omega/line.velocity)*line.length
        if np.any(propagation.real > 300):
            raise ValueError("Line attenuation is too large for this model")
        A = np.cosh(propagation)
        B = line.z0*np.sinh(propagation)
        C = np.sinh(propagation)/line.z0
        # Currents at both terminals point into the line.
        # Va - A Vb + B Ib = 0; Ia - C Vb + A Ib = 0.
        if a is not None:
            matrix[:,a,ia] += 1
            matrix[:,ia,a] += 1
        if b is not None:
            matrix[:,b,ib] += 1
            matrix[:,ia,b] -= A
            matrix[:,ib,b] -= C
        matrix[:,ia,ib] += B
        matrix[:,ib,ia] += 1
        matrix[:,ib,ib] += A
    drives = np.broadcast_to(incidence, (len(f), *incidence.shape))
    try:
        solution = np.linalg.solve(matrix, drives)
    except np.linalg.LinAlgError as error:
        raise ValueError("Line circuit is singular; check nodes and connections") from error
    return 2/z0*(incidence.T @ solution)-np.eye(len(port_nodes),dtype=complex)


def validate_network(elements, port_nodes):
    if len(port_nodes) < 1:
        raise ValueError("Add at least one port")
    if any(normalize_node(n) == GROUND for n in port_nodes):
        raise ValueError("A measurement port cannot use the ground node")
    if not elements:
        raise ValueError("Add at least one component")
    names = [e.name for e in elements]
    if len(names) != len(set(names)):
        raise ValueError("Component names must be unique")
    for e in elements:
        if isinstance(e, TransmissionLine):
            e.Validate()
        if e.kind not in ("R", "L", "C", "TL") or e.node_a == e.node_b:
            raise ValueError(f"Check the type and endpoints of {e.name}")
        if not e.node_a or not e.node_b or e.value <= 0 or not np.isfinite(e.value):
            raise ValueError(f"Check the nodes and value of {e.name}")
    # Floating subcircuits cannot be reduced to port voltages.
    adjacent = {}
    for e in elements:
        adjacent.setdefault(e.node_a, set()).add(e.node_b)
        adjacent.setdefault(e.node_b, set()).add(e.node_a)
    explored = set()
    for node in adjacent:
        if node in explored:
            continue
        stack, group = [node], set()
        while stack:
            current = stack.pop()
            if current in group:
                continue
            group.add(current)
            stack += list(adjacent[current] - group)
        explored |= group
        if not group.intersection(port_nodes):
            parts = ", ".join(e.name for e in elements
                              if e.node_a in group or e.node_b in group)
            loose = ", ".join(sorted(group - {GROUND}))
            raise ValueError(
                f"{parts} is connected to node(s) {loose}, but none is a port. "
                f"Edit a component node to {port_nodes[0]} (port 1), "
                "or add a port at that node.")


def s_parameters(frequency, elements, port_nodes, z0=50):
    """Nodal analysis with independent port terminations, even on shared nodes."""
    f = np.asarray(frequency, dtype=float)
    if f.ndim != 1 or len(f) < 1 or np.any(f <= 0) or not np.all(np.isfinite(f)):
        raise ValueError("Use at least one positive frequency sample")
    if z0 <= 0 or not np.isfinite(z0):
        raise ValueError("Z0 must be positive")
    elements = [e if isinstance(e, TransmissionLine) else
                Element.From_SI(e.name,e.kind,normalize_node(e.node_a),normalize_node(e.node_b),e.value,
                               e.display_unit) for e in elements]
    port_nodes = [normalize_node(n) for n in port_nodes]
    validate_network(elements, port_nodes)
    if any(isinstance(e, TransmissionLine) for e in elements):
        return _line_s_parameters(f, elements, port_nodes, z0)
    boundary = list(dict.fromkeys(port_nodes))
    nodes = boundary + sorted({n for e in elements
                                       for n in (e.node_a, e.node_b)
                                       if n != GROUND and n not in boundary})
    index = {name: i for i, name in enumerate(nodes)}
    y = np.zeros((len(f), len(nodes), len(nodes)), dtype=complex)
    omega = 2 * np.pi * f
    for e in elements:
        branch = ({"R": lambda: np.full(len(f), 1 / e.value),
                   "L": lambda: 1 / (1j * omega * e.value),
                   "C": lambda: 1j * omega * e.value})[e.kind]()
        a, b = index.get(e.node_a), index.get(e.node_b)
        if a is not None:
            y[:, a, a] += branch
        if b is not None:
            y[:, b, b] += branch
        if a is not None and b is not None:
            y[:, a, b] -= branch
            y[:, b, a] -= branch
    nb = len(boundary)
    if len(nodes) > nb:
        try:
            reduced = y[:, :nb, :nb] - y[:, :nb, nb:] @ np.linalg.solve(
                y[:, nb:, nb:], y[:, nb:, :nb])
        except np.linalg.LinAlgError as exc:
            raise ValueError("Internal node matrix is singular; check disconnected or floating parts") from exc
    else:
        reduced = y
    nport = len(port_nodes)
    incidence = np.zeros((nb,nport), dtype=complex)
    for j,node in enumerate(port_nodes):
        incidence[boundary.index(node),j] = 1
    # Each measurement port has its own Z0 termination, including ports that
    # share one electrical node with another measurement port.
    matched = reduced + incidence @ incidence.T / z0
    drives = np.broadcast_to(incidence,(len(f),nb,nport))
    try:
        voltages = np.linalg.solve(matched,drives)
        return 2 / z0 * (incidence.T @ voltages) - np.eye(nport,dtype=complex)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Could not convert circuit admittance to S-parameters") from exc


def smith_grid(ax, z0=50, *, impedance_labels=True):
    """Impedance Smith grid; plotted coordinates are reflection coefficients."""
    ax.add_patch(Circle((0, 0), 1, fill=False, lw=1.3, color="0.3"))
    x = np.linspace(-30, 30, 700)
    for r in (0, .2, .5, 1, 2, 5):
        gamma = (r + 1j*x - 1) / (r + 1j*x + 1)
        ax.plot(gamma.real, gamma.imag, color="0.86", lw=.7)
    resistance = np.linspace(0, 30, 700)
    for reactance in (.2, .5, 1, 2, 5):
        for sign in (-1, 1):
            gamma = (resistance + 1j*sign*reactance - 1) / (resistance + 1j*sign*reactance + 1)
            ax.plot(gamma.real, gamma.imag, color="0.86", lw=.7)
    if impedance_labels:
        # The coordinates are Γ (unitless); labels inside the chart show Z in Ω.
        for r in (0, .2, .5, 1, 2, 5):
            gx=(r-1)/(r+1)
            ax.text(gx,-.045,f"{r*z0:g}",ha="center",va="top",fontsize=7,color="0.35",
                    bbox=dict(facecolor="white",edgecolor="none",pad=.3))
        for xreact in (.5, 1, 2):
            for sign in (-1, 1):
                z=.15+1j*sign*xreact
                gamma=(z-1)/(z+1)
                ax.text(gamma.real,gamma.imag,f"{sign*xreact*z0:+g}j Ω",
                        ha="center",va="bottom" if sign>0 else "top",fontsize=6.5,
                        color="0.43",bbox=dict(facecolor="white",edgecolor="none",pad=.2))
        ax.text(.035,.055,f"{z0:g} + j0 Ω",color="#a33a1b",fontsize=8,
                bbox=dict(facecolor="white",edgecolor="none",pad=1))
    ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal",
           xlabel="Real part of Γ = Sii (unitless)",
           ylabel="Imaginary part of Γ = Sii (unitless)",
           title=f"Smith chart · impedance labels in Ω · Z0 = {z0:g} Ω")


def export_csv(filename, frequency, s):
    with open(filename, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        n = s.shape[1]
        writer.writerow(["frequency_Hz"] + [field for i in range(n) for j in range(n)
                        for field in (f"S{i+1}{j+1}_real", f"S{i+1}{j+1}_imag",
                                      f"S{i+1}{j+1}_dB", f"S{i+1}{j+1}_phase_deg")])
        for k, f in enumerate(frequency):
            writer.writerow([f] + [field for i in range(n) for j in range(n)
                               for field in (s[k,i,j].real, s[k,i,j].imag,
                                             20*np.log10(max(abs(s[k,i,j]), 1e-300)),
                                             np.angle(s[k,i,j], deg=True))])


def export_touchstone(filename, frequency, s, z0):
    """Touchstone v1, RI format, with output index varying fastest."""
    with open(filename, "w", encoding="utf-8") as stream:
        stream.write(f"! RF Circuit Studio ideal RLC network, {s.shape[1]} ports\n")
        stream.write(f"# Hz S RI R {z0:.12g}\n")
        for f, matrix in zip(frequency, s):
            values = " ".join(f"{matrix[i,j].real:.12g} {matrix[i,j].imag:.12g}"
                              for j in range(s.shape[1]) for i in range(s.shape[1]))
            stream.write(f"{f:.12g} {values}\n")


def example_network():
    # Symmetric 3-port resistive splitter, with a small junction capacitance.
    return ([Element("R1", "R", "P1", "J", 50 / 3),
             Element("R2", "R", "P2", "J", 50 / 3),
             Element("R3", "R", "P3", "J", 50 / 3),
             Element("C1", "C", "J", GROUND, 5e-12)],
            ["P1", "P2", "P3"])


def capacitor_example():
    """One capacitor from port 1 to ground: only S11 exists."""
    return [Element("C1", "C", "P1", GROUND, 10e-12)], ["P1"]


def project_data(elements, ports, settings, title="Untitled circuit"):
    return {"format": "rf-circuit-studio", "version": 1,
            "title": title,
            "ports": list(ports),
            "elements": [dict(name=e.name, kind=e.kind, node_a=e.node_a,
                              node_b=e.node_b, value=e.value,
                              display_unit=e.display_unit) for e in elements],
            "settings": settings}


def terminate_shared_ports(elements, ports, z0):
    """Retain the first port on each node and terminate each duplicate in Z0.

    This preserves the selected S-parameter submatrix of a Qucs project with
    several physical ports attached to one circuit node.
    """
    unique, seen = [], set()
    converted = list(elements)
    names = {e.name for e in converted}
    for node in ports:
        if node not in seen:
            unique.append(node)
            seen.add(node)
            continue
        number = 1
        while f"R_term_{node}_{number}" in names:
            number += 1
        name = f"R_term_{node}_{number}"
        names.add(name)
        converted.append(Element(name, "R", node, GROUND, z0, "Ω"))
    return converted, unique


def qucs_channel_coupling(s, elements, ports, z0):
    """Return Qucs S14/S23 for four shared ports or the equivalent two-port.

    A duplicated matched port becomes a Z0 shunt in the two-port model.  The
    four-port diagonal Sii has the same value as the corresponding two-port
    Sii, while the transfer to its co-located port is 1 + Sii.
    """
    if len(ports)==4 and ports == [ports[0],ports[1],ports[1],ports[0]]:
        return s[:,0,3], s[:,1,2]
    if len(ports) != 2 or s.shape[1:] != (2, 2):
        raise ValueError("Qucs channel coupling needs P1/P2/P2/P1 shared ports "
                         "or their matched two-port equivalent")
    for node in ports:
        matches = [e for e in elements if e.kind == "R"
                   and e.value == z0 and {e.node_a, e.node_b} == {node, GROUND}
                   and (e.name.startswith("R_terminated_Qucs_port_")
                        or e.name.startswith("R_term_"))]
        if len(matches) != 1:
            raise ValueError("Load the Qucs-matched two-port JSON with one "
                             f"extra {z0:g} Ω port termination at each node")
    return 1 + s[:, 0, 0], 1 + s[:, 1, 1]


def refine_qucs_sweep(f, s, elements, ports, z0):
    """Add the same resonance samples to all views and exported S-parameters."""
    try:
        channels = qucs_channel_coupling(s, elements, ports, z0)
    except ValueError:
        return f, s, None
    halfwidth = max(10e6, (f[-1] - f[0]) * .025)
    local_grids = []
    for channel in channels:
        center = f[np.argmin(abs(channel))]
        local_grids.append(np.linspace(max(f[0], center-halfwidth),
                                       min(f[-1], center+halfwidth), 12001))
    refined_f = np.unique(np.concatenate((f, *local_grids)))
    refined_s = s_parameters(refined_f, elements, ports, z0)
    windows = []
    for local_f in local_grids:
        indices = np.searchsorted(refined_f, local_f)
        windows.append((local_f, refined_s[indices]))
    return refined_f, refined_s, windows


def port_input_impedance(s, z0, port):
    """Input impedance with every other measurement port terminated in Z0."""
    gamma = s[:, port, port]
    with np.errstate(divide="ignore", invalid="ignore"):
        return z0 * (1 + gamma) / (1 - gamma)


def bandlimited_impulse(frequency, response, samples=8192):
    """Complex baseband impulse response from a finite positive-frequency band.

    The Hann taper reduces sharp band-edge ringing. The result is an envelope,
    not a DC-to-infinity transient or a step response.
    """
    f = np.asarray(frequency, dtype=float)
    if f.ndim != 1 or len(f) < 3 or f[-1] <= f[0]:
        raise ValueError("Run a frequency sweep before viewing time response")
    uniform_f = np.linspace(f[0], f[-1], samples, endpoint=False)
    values = (np.interp(uniform_f, f, np.real(response)) +
              1j*np.interp(uniform_f, f, np.imag(response)))
    bandwidth = f[-1] - f[0]
    df = bandwidth / samples
    spectrum = values * np.hanning(samples)
    time = (np.arange(samples) - samples//2) / bandwidth
    envelope = np.fft.fftshift(np.fft.ifft(spectrum)) * bandwidth
    envelope *= np.exp(-1j * np.pi * bandwidth * time)
    peak = np.max(np.abs(envelope))
    if peak > 0:
        envelope /= peak
    return time, envelope


def read_project(data):
    if data.get("format") != "rf-circuit-studio" or data.get("version") != 1:
        raise ValueError("Not an RF Circuit Studio project file")
    elements = [Element.From_SI(item["name"], item["kind"], normalize_node(item["node_a"]),
                        normalize_node(item["node_b"]), float(item["value"]),
                        item.get("display_unit", "")) for item in data["elements"]]
    ports = [normalize_node(n) for n in data["ports"]]
    settings = data["settings"]
    start, stop, z0 = (parse_value(settings[key]) for key in ("start", "stop", "z0"))
    points = int(settings["points"])
    if stop <= start or not 3 <= points <= 20000:
        raise ValueError("Invalid saved frequency sweep")
    validate_network(elements, ports)
    for name, limits in settings.get("tuning_ranges", {}).items():
        if not isinstance(name, str) or len(limits) != 2 or not (0 < float(limits[0]) < float(limits[1])):
            raise ValueError("Invalid saved tuning range")
    s_parameters(np.array([start, (start+stop)/2, stop]), elements, ports, z0)
    return elements, ports, settings


class RFCircuit:
    """RLC and uniform transmission-line networks for Jupyter notebooks."""

    def __init__(self, elements, ports, z0=50.0, title="RF circuit", layout=None, *, verbose=True):
        self.elements = list(elements)
        # Separate physical port labels from electrical connection nodes.
        # Legacy node-only lists retain their original ordering and loading.
        self.port_definitions = []
        for index, entry in enumerate(ports, start=1):
            if isinstance(entry, str):
                label, node = f"P{index}", entry
            elif isinstance(entry, (tuple, list)) and len(entry) == 2:
                label, node = entry
                if not isinstance(label, str) or not isinstance(node, str):
                    raise ValueError("Port labels and node names must be strings")
            else:
                raise ValueError("Define each port as ('P1', 'P1') or a node name")
            label, node = label.strip(), normalize_node(node)
            if not label or not node:
                raise ValueError("Port labels and node names cannot be empty")
            self.port_definitions.append((label, node))
        self.port_labels = [label for label, _ in self.port_definitions]
        if len(set(self.port_labels)) != len(self.port_labels):
            raise ValueError("Physical port labels must be unique; connection nodes may repeat")
        self.ports = [node for _, node in self.port_definitions]
        self.z0 = float(z0)
        if not np.isfinite(self.z0) or self.z0 <= 0:
            raise ValueError("Reference impedance must be positive and finite")
        self.title = str(title)
        self.layout_positions = dict(layout or {})
        self.circuit_title = SimpleNamespace(get=lambda: self.title)
        self.frequency = None
        self.s = None
        self.figure = None
        self.verbose = bool(verbose)
        validate_network(self.elements, self.ports)
        if self.verbose:
            self.Print_Info()

    def Print_Info(self):
        """Print the circuit title, physical ports, nodes and reference impedance."""
        print(self.title)
        for number, (label, node) in enumerate(self.port_definitions, start=1):
            print(f"Port {number} ({label}) -> node {node}; reference node {GROUND}")
        print("All electrical nodes:", self.nodes)
        print("Reference impedance:", self.z0, "Ω")

    @property
    def port_nodes(self):
        """Electrical signal node for each physical port, in S-matrix order."""
        return list(self.ports)

    @property
    def nodes(self):
        """All named electrical nodes, including internal junctions and ground."""
        return list(dict.fromkeys([*self.ports,
                    *(node for e in self.elements for node in (e.node_a, e.node_b))]))

    @classmethod
    def From_Json(cls, filename, *, verbose=True):
        """Load circuit parameters; equivalent to Load_Parameters."""
        return cls.Load_Parameters(filename, verbose=verbose)

    @classmethod
    def Load_Parameters(cls, filename, *, verbose=True):
        """Create a circuit from saved parameters or a legacy Studio project."""
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Circuit JSON must contain an object")
        if data.get("format") == "pyor-rf-parameters":
            if data.get("version") != 1:
                raise ValueError("Unsupported RF parameter file version")
            try:
                elements = [_load_component(item) for item in data["components"]]
                ports = [(item["label"], item["node"]) for item in data["ports"]]
                circuit = cls(elements, ports, data["z0"], data.get("title", "RF circuit"),
                              data.get("layout", {}), verbose=verbose)
                settings = data.get("sweep", {})
                if not isinstance(settings, dict):
                    raise ValueError("Saved sweep must be an object")
                circuit.settings = settings
                return circuit
            except (KeyError, TypeError) as error:
                raise ValueError(f"Invalid RF parameter file: {error}") from error
        elements, ports, settings = read_project(data)
        circuit = cls(elements, ports, parse_value(settings["z0"]),
                      data.get("title", "RF circuit"), data.get("layout", {}), verbose=verbose)
        circuit.settings = settings
        return circuit

    def Save_Parameters(self, filename):
        """Save component values/units, named ports, title, layout and sweep to JSON."""
        components = []
        for element in self.elements:
            if isinstance(element, TransmissionLine):
                components.append(element._parameters())
                continue
            unit = element.display_unit or {"R":"Ω", "L":"H", "C":"F"}[element.kind]
            components.append(dict(name=element.name, kind=element.kind,
                              node_a=element.node_a, node_b=element.node_b,
                              value=element.value / UNIT_SCALE[unit], unit=unit))
        data = dict(format="pyor-rf-parameters", version=1, title=self.title,
                    z0=self.z0, ports=[dict(label=label, node=node)
                                      for label,node in self.port_definitions],
                    components=components, layout=self.layout_positions,
                    sweep=getattr(self, "settings", {}))
        path = Path(filename)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
                        encoding="utf-8")
        return path

    def Simulate(self, frequency=None, *, start=None, stop=None, points=None,
                 logarithmic=None, refine=None):
        """Calculate S[f,output,input]; frequency is in Hz, component values SI.

        If frequency is omitted, use the sweep stored in the loaded JSON file.
        Explicit start/stop/points override those stored settings.
        """
        saved = getattr(self, "settings", {})
        refine = bool(saved.get("refine", False) if refine is None else refine)
        if frequency is None:
            start = parse_value(start if start is not None else saved.get("start", "100MHz"))
            stop = parse_value(stop if stop is not None else saved.get("stop", "550MHz"))
            points = int(points if points is not None else saved.get("points", 501))
            logarithmic = bool(logarithmic if logarithmic is not None
                               else saved.get("log", False))
            if stop <= start or points < 3:
                raise ValueError("Use start < stop and at least 3 frequency points")
            frequency = (np.geomspace if logarithmic else np.linspace)(start, stop, points)
        frequency = np.asarray(frequency, dtype=float)
        if (frequency.ndim != 1 or len(frequency) < 3 or
            not np.all(np.isfinite(frequency)) or np.any(frequency <= 0) or
            np.any(np.diff(frequency) <= 0)):
            raise ValueError("Frequencies must be strictly increasing, with at least 3 samples")
        self.settings = dict(start=float(frequency[0]), stop=float(frequency[-1]),
                             points=len(frequency), log=bool(logarithmic), refine=refine)
        s = s_parameters(frequency, self.elements, self.ports, self.z0)
        if refine:
            frequency, s, _ = refine_qucs_sweep(frequency, s, self.elements,
                                                 self.ports, self.z0)
        self.frequency, self.s = frequency, s
        if self.verbose:
            print(f"{len(frequency):,} frequency samples; S shape = {s.shape}")
        return frequency, s

    def _result(self):
        if self.s is None:
            self.Simulate()
        return self.frequency, self.s

    @staticmethod
    def Show():
        """Display plots in Jupyter without importing Matplotlib in the notebook."""
        plt.show()

    def Trace_Minimum(self, trace="S11"):
        """Return the frequency and complex value of the lowest |Sij| sample."""
        f,s = self._result()
        match = re.fullmatch(r"S([1-9])([1-9])",str(trace).upper())
        if not match:
            raise ValueError("Use an Sij trace, such as S11 or S23")
        i,j = int(match[1])-1,int(match[2])-1
        if i >= len(self.ports) or j >= len(self.ports):
            raise ValueError("Unknown port number")
        index = int(np.argmin(abs(s[:,i,j])))
        return {"index":index, "frequency_Hz":float(f[index]),
                "frequency_MHz":float(f[index]/1e6), "S":complex(s[index,i,j])}

    def Impedance_At(self, frequency, port=1, unit="MHz"):
        """Interpolate complex input impedance at a frequency on the sweep."""
        f,_ = self._result()
        scales={"Hz":1,"kHz":1e3,"MHz":1e6,"GHz":1e9}
        if unit not in scales:
            raise ValueError("Frequency unit must be Hz, kHz, MHz or GHz")
        target=float(frequency)*scales[unit]
        if not np.isfinite(target) or not f[0] <= target <= f[-1]:
            raise ValueError("Frequency must lie within the simulated sweep")
        z=self.Input_Impedance(port)
        return complex(np.interp(target,f,z.real),np.interp(target,f,z.imag))

    def Input_Impedance(self, port=1):
        """Zin in ohms at physical port (1-based); all other ports at Z0."""
        _, s = self._result()
        if not 1 <= port <= len(self.ports):
            raise ValueError("Unknown port number")
        return port_input_impedance(s, self.z0, port-1)

    def Plot_Sparameter(self, traces=("S11",), *, phase=False, ax=None, xlim=None, ylim=None,
                        show=True):
        f,s = self._result()
        if isinstance(traces, str):
            traces = (traces,)
        if ax is None:
            _,ax=plt.subplots(figsize=(9,4))
        for trace in traces:
            match = re.fullmatch(r"S(\d+)(\d+)",trace.upper())
            if not match:
                raise ValueError(f"Expected Sij trace, got {trace!r}")
            i,j=int(match[1])-1,int(match[2])-1
            if not (0 <= i < len(self.ports) and 0 <= j < len(self.ports)):
                raise ValueError(f"Unknown trace {trace}")
            value=s[:,i,j]
            ordinate=(np.angle(value, deg=True) if phase else
                      20*np.log10(np.maximum(abs(value),1e-300)))
            ax.plot(f/1e6,ordinate,label=trace.upper())
        ax.set(xlabel="Frequency (MHz)",ylabel="Phase (degrees)" if phase else "Magnitude (dB)",
               title=self.title)
        ax.grid(alpha=.3);ax.legend()
        if xlim is not None: ax.set_xlim(xlim)
        if phase:
            ax.set_ylim((-180, 180) if ylim is None else ylim)
        elif ylim is not None:
            ax.set_ylim(ylim)
        if show: self.Show()
        return ax

    def Plot_Impedance(self, port=1, *, absolute=False, ax=None, xlim=None,
                       ylim=None, ylim_real=None, ylim_imag=None, show=True):
        f,_ = self._result()
        z = self.Input_Impedance(port)
        if ax is None:
            _,ax=plt.subplots(2,1,figsize=(9,6),sharex=True)
        r,x=z.real,z.imag
        ax[0].plot(f/1e6,np.abs(r) if absolute else r,label=f"Port {port}")
        ax[1].plot(f/1e6,np.abs(x) if absolute else x,label=f"Port {port}")
        ax[0].axhline(self.z0,color=".5",ls="--",lw=1,label=f"{self.z0:g} Ω reference")
        ax[1].axhline(0,color=".5",ls="--",lw=1)
        ax[0].set(title=f"Input impedance from S{port}{port}; other ports at {self.z0:g} Ω",
                  ylabel="|Real(Z)| (Ω)" if absolute else "Real(Z) (Ω)")
        ax[1].set(xlabel="Frequency (MHz)",
                  ylabel="|Imag(Z)| (Ω)" if absolute else "Imag(Z) (Ω)")
        for axis in ax:
            axis.grid(alpha=.3);axis.legend()
            if xlim is not None: axis.set_xlim(xlim)
            if ylim is not None: axis.set_ylim(ylim)
        if ylim_real is not None: ax[0].set_ylim(ylim_real)
        if ylim_imag is not None: ax[1].set_ylim(ylim_imag)
        if show: self.Show()
        return ax

    def Plot_Smith(self, trace="S11", *, ax=None, xlim=None, ylim=None,
                   frequency_limits=None, show=True):
        """Plot a complex Sij trace; impedance labels apply only to Sii."""
        if not isinstance(trace, str):
            raise ValueError("Use a trace name such as S11 or S14")
        trace = trace.strip().upper()
        match = re.fullmatch(r"S([1-9])([1-9])", trace)
        if not match:
            raise ValueError("Use a trace name such as S11 or S14")
        i, j = int(match[1])-1, int(match[2])-1
        if i >= len(self.ports) or j >= len(self.ports):
            raise ValueError(f"Unknown trace {trace}")
        f, s = self._result()
        indices = np.arange(len(f))
        if frequency_limits is not None:
            low, high = map(float, frequency_limits)
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                raise ValueError("Use increasing finite frequency limits in MHz")
            indices = np.flatnonzero((f/1e6 >= low) & (f/1e6 <= high))
            if not len(indices):
                raise ValueError("No simulation samples inside the selected frequency window")
        reflection = i == j
        if ax is None:
            _, ax = plt.subplots(figsize=(7,7))
        smith_grid(ax, self.z0, impedance_labels=reflection)
        value = s[indices,i,j]
        ax.plot(value.real, value.imag, label=trace)
        best = int(np.argmin(abs(value)))
        sample = indices[best]
        ax.plot(value[best].real, value[best].imag, "o", color="#c24b24")
        if reflection:
            z = self.Input_Impedance(i+1)[sample]
            annotation = (f"Closest match: {f[sample]/1e6:.6g} MHz\n"
                          f"Zin={z.real:.4g}{z.imag:+.4g}j Ω")
            title = f"{trace} Smith chart; impedance labels in Ω; Z0 = {self.z0:g} Ω"
        else:
            annotation = f"Minimum |{trace}|: {f[sample]/1e6:.6g} MHz"
            title = f"{trace} transmission on Smith grid; dimensionless S-parameter"
        ax.set(xlabel=f"Real({trace}) (unitless)",
               ylabel=f"Imag({trace}) (unitless)", title=title)
        ax.text(.02,.02,annotation, transform=ax.transAxes,fontsize=8,
                bbox=dict(facecolor="white",edgecolor=".8",pad=4))
        ax.legend(loc="upper right")
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if show: self.Show()
        return ax

    def _level_crossings(self, trace, window, *, relative=False):
        """Find -3 dB crossings in MHz and refine them by circuit evaluation."""
        f, s = self._result()
        i, j = int(trace[1])-1, int(trace[2])-1
        low, high = np.asarray(window, dtype=float)*1e6
        mask = (f > low) & (f < high)
        frequencies = np.r_[low, f[mask], high]
        endpoints = s_parameters(np.array([low, high]), self.elements, self.ports, self.z0)
        values = np.r_[endpoints[0,i,j], s[mask,i,j], endpoints[1,i,j]]
        db = 20*np.log10(np.maximum(abs(values), 1e-300))
        target = float(np.max(db)-3 if relative else -3)
        difference = db-target
        roots = []

        def Magnitude(hz):
            value = s_parameters(np.array([hz]), self.elements, self.ports, self.z0)[0,i,j]
            return float(20*np.log10(max(abs(value), 1e-300)))

        for index in range(len(frequencies)-1):
            a, b = frequencies[index:index+2]
            da, db_value = difference[index:index+2]
            if da == 0:
                roots.append(float(a/1e6))
            if da*db_value < 0:
                # Bisection evaluates S at each candidate frequency directly.
                for iteration in range(60):
                    middle = (a+b)/2
                    dm = Magnitude(middle)-target
                    if abs(dm) < 1e-8 or b-a < 1e-4:
                        break
                    if da*dm <= 0:
                        b = middle
                    else:
                        a, da = middle, dm
                roots.append(float(middle/1e6))
        if difference[-1] == 0:
            roots.append(float(high/1e6))
        roots = sorted(set(roots))
        if len(roots) > 2:
            # Prefer the two crossings around the deepest notch or highest peak.
            if difference[0] >= 0 and difference[-1] >= 0:
                center = frequencies[int(np.argmin(difference))]/1e6
            else:
                center = frequencies[int(np.argmax(difference))]/1e6
            left = [root for root in roots if root < center]
            right = [root for root in roots if root > center]
            roots = [left[-1], right[0]] if left and right else [roots[0], roots[-1]]
        return roots, target

    def Plot_Interactive(self, traces=("S14", "S23", "S21", "S12"), *,
                         xlim=None, ylim=(-160, 5), component_limits=None,
                         points=1001, refine=True, markers=None):
        """Display Jupyter sliders for component tuning and plot limits.

        Component ranges are (min, max[, step]) in each element's display
        unit. Defaults span 50 to 150 percent of the initial value. Changes
        update this circuit and its simulation; Reset restores initial values.
        The level button finds absolute -3 dB crossings or 3 dB below the
        peak within the visible frequency window.
        Two marker sliders report magnitude (dB), phase wrapped to -180 through 180 degrees
        and input impedance (ohms). Marker frequencies are in MHz. Impedance
        comes from Spp at the selected trace's output port. Transmission Sij is never converted to input impedance.
        A running Jupyter kernel and ipywidgets are required.
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as error:
            raise ImportError("Install widgets in a notebook cell with "
                              "%pip install ipywidgets, then restart the kernel") from error
        if isinstance(traces, str):
            traces = (traces,)
        traces = tuple(dict.fromkeys(str(t).strip().upper() for t in traces))
        if not traces:
            raise ValueError("Select at least one S-parameter trace")
        for trace in traces:
            match = re.fullmatch(r"S([1-9])([1-9])", trace)
            if not match or max(int(match[1]), int(match[2])) > len(self.ports):
                raise ValueError(f"Unknown trace {trace}")
        if int(points) != points or points < 3:
            raise ValueError("Use at least 3 simulation points")
        saved = dict(getattr(self, "settings", {}))
        start = parse_value(saved.get("start", "100MHz"))
        stop = parse_value(saved.get("stop", "550MHz"))
        logarithmic = bool(saved.get("log", True))
        lo, hi = start/1e6, stop/1e6
        xlim = (lo, hi) if xlim is None else tuple(xlim)
        if not lo <= xlim[0] < xlim[1] <= hi:
            raise ValueError("Frequency limits must lie inside the sweep, in MHz")
        automatic = ylim is None
        ylim = (-160, 5) if ylim is None else tuple(ylim)
        if len(ylim) != 2 or not np.all(np.isfinite(ylim)) or ylim[0] >= ylim[1]:
            raise ValueError("Use increasing finite magnitude limits, in dB")
        if markers is None:
            markers = (xlim[0]+(xlim[1]-xlim[0])*.25,
                       xlim[0]+(xlim[1]-xlim[0])*.75)
        markers = tuple(markers)
        if (len(markers) != 2 or not np.all(np.isfinite(markers)) or
            any(not xlim[0] <= value <= xlim[1] for value in markers)):
            raise ValueError("Use two marker frequencies inside xlim, in MHz")
        component_limits = {} if component_limits is None else dict(component_limits)
        unknown = set(component_limits) - {e.name for e in self.elements}
        if unknown:
            raise ValueError(f"Unknown component names: {sorted(unknown)}")
        style = {"description_width": "180px"}
        layout = widgets.Layout(width="95%")
        initial = [e.value for e in self.elements]
        sliders = []
        scales = []
        line_controls = {}
        for element in self.elements:
            unit = element.display_unit or {"R":"Ω", "L":"H", "C":"F"}[element.kind]
            scale = UNIT_SCALE[unit]
            value = element.value/scale
            bounds = component_limits.get(element.name, (value*.5, value*1.5))
            if len(bounds) not in (2, 3):
                raise ValueError("Component ranges need min, max and optional step")
            minimum, maximum = map(float, bounds[:2])
            step = float(bounds[2]) if len(bounds) == 3 else (maximum-minimum)/200
            if (not np.all(np.isfinite([minimum, maximum, step])) or
                not 0 < minimum <= value <= maximum or minimum >= maximum or step <= 0):
                raise ValueError(f"Invalid slider range for {element.name}")
            sliders.append(widgets.FloatSlider(value=value, min=minimum, max=maximum,
                step=step, description=f"{element.name} ({unit})", readout_format=".6g",
                continuous_update=False, style=style, layout=layout))
            scales.append(scale)
            if isinstance(element, TransmissionLine):
                max_er = max(25, element.epsilon_r*1.5)
                def Field(value, minimum, maximum, description):
                    return widgets.FloatSlider(value=value, min=minimum, max=maximum,
                        step=(maximum-minimum)/500, description=description,
                        readout_format=".6g", continuous_update=False, style=style, layout=layout)
                line_controls[element.name] = dict(length=sliders[-1],
                    z0=Field(element.z0,element.z0*.5,element.z0*1.5,f"{element.name} Z0 (Ω)"),
                    velocity_factor=Field(element.velocity_factor,1/np.sqrt(max_er),1,f"{element.name} VF"),
                    epsilon_r=Field(element.epsilon_r,1,max_er,f"{element.name} epsilon_r"),
                    attenuation_db_per_m=Field(element.attenuation_db_per_m,0,
                        max(2,element.attenuation_db_per_m*2),f"{element.name} loss (dB/m)"))
        line_initial = {e.name:(e.z0,e.velocity_factor,e.attenuation_db_per_m)
                        for e in self.elements if isinstance(e,TransmissionLine)}
        trace_control = widgets.Dropdown(options=traces, value=traces[0], description="Trace")
        frequency_control = widgets.FloatRangeSlider(value=xlim, min=lo, max=hi,
            step=(hi-lo)/10000, description="Frequency (MHz)", readout_format=".3f",
            continuous_update=False, style=style, layout=layout)
        magnitude_control = widgets.FloatRangeSlider(value=ylim,
            min=min(-200, ylim[0]), max=max(10, ylim[1]), step=1,
            description="Magnitude (dB)", continuous_update=False, style=style, layout=layout)
        auto_control = widgets.Checkbox(value=automatic, description="Automatic dB limits")
        marker_controls = [widgets.FloatSlider(value=value, min=xlim[0], max=xlim[1],
            step=(xlim[1]-xlim[0])/10000, description=f"Marker {index} (MHz)",
            readout_format=".6f", continuous_update=False, style=style, layout=layout)
            for index, value in enumerate(markers, start=1)]
        level_reference = widgets.Dropdown(
            options=[("Absolute -3 dB", False), ("3 dB below peak in window", True)],
            value=False, description="Level reference", style=style, layout=layout)
        level_button = widgets.Button(description="Set markers to -3 dB",
                                      layout=widgets.Layout(width="200px"))
        reset_button = widgets.Button(description="Reset")
        status = widgets.Label()
        output = widgets.Output()
        state = {"busy": False, "fig": None, "error": None}

        def Sync_Markers(window):
            for control in marker_controls:
                # Expand before moving a marker, then restrict to the new window.
                control.min = min(control.min, window[0])
                control.max = max(control.max, window[1])
                control.value = min(max(control.value, window[0]), window[1])
                control.min, control.max = window
                control.step = (window[1]-window[0])/10000

        def Update(change=None, *, simulate=False):
            if state["busy"]:
                return
            state["busy"] = True
            status.value = "Updating..."
            old_values = [e.value for e in self.elements]
            old_lines = {e.name:(e.z0,e.velocity_factor,e.attenuation_db_per_m)
                         for e in self.elements if isinstance(e,TransmissionLine)}
            old_frequency, old_s = self.frequency, self.s
            old_settings = dict(getattr(self, "settings", {}))
            previous_verbose = self.verbose
            try:
                if simulate or self.s is None:
                    for element, slider, scale in zip(self.elements, sliders, scales):
                        element.value = slider.value*scale
                        if isinstance(element, TransmissionLine):
                            controls = line_controls[element.name]
                            for field in ("z0", "velocity_factor", "attenuation_db_per_m"):
                                setattr(element, field, controls[field].value)
                    self.verbose = False
                    self.Simulate(start=start, stop=stop, points=int(points),
                                  logarithmic=logarithmic, refine=refine)
                trace = trace_control.value
                window = frequency_control.value
                Sync_Markers(window)
                fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
                state["fig"] = fig
                self.Plot_Sparameter(trace, ax=axes[0], xlim=window,
                    ylim=None if auto_control.value else magnitude_control.value, show=False)
                self.Plot_Sparameter(trace, phase=True, ax=axes[1], xlim=window, show=False)
                self.Plot_Smith(trace, ax=axes[2], frequency_limits=window, show=False)
                axes[0].set_title(f"{trace} magnitude")
                axes[1].set_title(f"{trace} phase")
                axes[2].set_title(f"{trace} Smith chart" if trace[1] == trace[2]
                                  else f"{trace} transmission on Smith grid")
                # Evaluate the circuit at the marker frequencies directly, so
                # narrow resonances do not depend on sweep interpolation.
                marker_frequencies = np.array([control.value for control in marker_controls])
                marker_s = s_parameters(marker_frequencies*1e6,
                                        self.elements, self.ports, self.z0)
                i, j = int(trace[1])-1, int(trace[2])-1
                port = i+1
                marker_values = marker_s[:,i,j]
                marker_db = 20*np.log10(np.maximum(abs(marker_values), 1e-300))
                marker_phase = np.angle(marker_values, deg=True)
                gamma = marker_s[:,port-1,port-1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    marker_z = self.z0*(1+gamma)/(1-gamma)
                marker_results = []
                for index, (mhz, db, phase, value, z) in enumerate(zip(
                    marker_frequencies, marker_db, marker_phase, marker_values, marker_z), start=1):
                    color = ("#b22222", "#176b35")[index-1]
                    symbol = "s"
                    for axis, ordinate in ((axes[0], db), (axes[1], phase)):
                        axis.axvline(mhz, color=color, ls="--", lw=.9, alpha=.7)
                        axis.plot(mhz, ordinate, marker=symbol, color=color,
                                  ls="none", label=f"M{index}", zorder=6)
                    axes[2].plot(value.real, value.imag, marker=symbol, color=color,
                                 ls="none", label=f"M{index}", zorder=6)
                    position = -.23-(index-1)*.09
                    common = f"M{index}: {mhz:.6f} MHz"
                    axes[0].text(.01, position, f"{common}; {db:.6g} dB",
                                 transform=axes[0].transAxes, color=color, fontsize=10)
                    axes[1].text(.01, position, f"{common}; {phase:.6g} degrees",
                                 transform=axes[1].transAxes, color=color, fontsize=10)
                    impedance_text = (f"{z.real:.6g} {z.imag:+.6g}j Ω"
                                      if np.isfinite(z) else "infinite impedance (open circuit)")
                    axes[2].text(.01, position, f"M{index}: R+jX = {impedance_text}",
                                 transform=axes[2].transAxes, color=color, fontsize=10)
                    marker_results.append(dict(marker=index, frequency_MHz=float(mhz),
                        trace=trace, magnitude_dB=float(db), phase_degrees=float(phase),
                        impedance_port=port, impedance_trace=f"S{port}{port}",
                        impedance_ohm=complex(z)))
                axes[2].text(.01, -.42,
                    f"Zin at {self.port_labels[port-1]} from S{port}{port}; other ports at {self.z0:g} Ω",
                    transform=axes[2].transAxes, fontsize=9)
                for axis in axes:
                    axis.legend(loc="upper right", fontsize=9)
                state["markers"] = marker_results
                fig.suptitle(self.title, fontweight="bold")
                fig.tight_layout()
                with output:
                    output.clear_output(wait=True)
                    display(fig)
                plt.close(fig)
                state["error"] = None
                status.value = (f"{len(self.frequency):,} samples; "
                                "component values are stored in circuit")
            except Exception as error:
                for element, value in zip(self.elements, old_values):
                    element.value = value
                    if isinstance(element, TransmissionLine):
                        element.z0, element.velocity_factor, element.attenuation_db_per_m = old_lines[element.name]
                self.frequency, self.s, self.settings = old_frequency, old_s, old_settings
                if state["fig"] is not None:
                    plt.close(state["fig"])
                state["error"] = error
                status.value = f"Update failed: {error}"
            finally:
                self.verbose = previous_verbose
                state["busy"] = False

        def Tune(change):
            Update(change, simulate=True)

        def Set_Level_Markers(button):
            if state["busy"]:
                return
            try:
                roots, target = self._level_crossings(trace_control.value,
                    frequency_control.value, relative=level_reference.value)
                if not roots:
                    status.value = (f"No {target:.6g} dB crossing in this frequency window. "
                                    "Try a wider window or 3 dB below peak.")
                    return
                state["busy"] = True
                try:
                    for control, frequency in zip(marker_controls, roots):
                        control.value = frequency
                finally:
                    state["busy"] = False
                Update()
                if state["error"] is None:
                    state["level_target_dB"] = target
                    status.value = (f"Markers at {target:.6g} dB; separation "
                                    f"{abs(roots[-1]-roots[0]):.6g} MHz" if len(roots) == 2
                                    else f"One {target:.6g} dB crossing: M1 updated; M2 unchanged")
            except Exception as error:
                state["busy"] = False
                status.value = f"Could not set level markers: {error}"

        def Reset(button):
            state["busy"] = True
            try:
                for slider, value, scale in zip(sliders, initial, scales):
                    slider.value = value/scale
                for name, (zc, vf, loss) in line_initial.items():
                    controls = line_controls[name]
                    controls["z0"].value = zc
                    controls["velocity_factor"].value = vf
                    controls["epsilon_r"].value = 1/vf**2
                    controls["attenuation_db_per_m"].value = loss
                frequency_control.value = xlim
                magnitude_control.value = ylim
                auto_control.value = automatic
                trace_control.value = traces[0]
                level_reference.value = False
                Sync_Markers(xlim)
                for control, value in zip(marker_controls, markers):
                    control.value = value
            finally:
                state["busy"] = False
            Update(simulate=True)

        for slider in sliders:
            slider.observe(Tune, names="value")
        def Tune_Line(change):
            if state["busy"]:
                return
            for controls in line_controls.values():
                vf, er = controls["velocity_factor"], controls["epsilon_r"]
                if change["owner"] is vf or change["owner"] is er:
                    state["busy"] = True
                    try:
                        if change["owner"] is vf: er.value = 1/vf.value**2
                        else: vf.value = 1/np.sqrt(er.value)
                    finally:
                        state["busy"] = False
                    break
            Tune(change)

        for controls in line_controls.values():
            for field in ("z0", "velocity_factor", "epsilon_r", "attenuation_db_per_m"):
                controls[field].observe(Tune_Line, names="value")
        for control in (trace_control, frequency_control, magnitude_control, auto_control,
                        *marker_controls):
            control.observe(Update, names="value")
        reset_button.on_click(Reset)
        level_button.on_click(Set_Level_Markers)
        tuning_controls = []
        for element, slider in zip(self.elements, sliders):
            tuning_controls.append(slider)
            if isinstance(element, TransmissionLine):
                tuning_controls.extend(line_controls[element.name][field] for field in
                    ("z0", "velocity_factor", "epsilon_r", "attenuation_db_per_m"))
        tuning = widgets.Accordion(children=[widgets.VBox(tuning_controls)])
        tuning.set_title(0, "Component values")
        tuning.selected_index = None
        panel = widgets.VBox([widgets.HBox([trace_control, reset_button]),
            frequency_control, magnitude_control, auto_control,
            *marker_controls, level_reference, level_button, tuning, status, output])
        # Retain controls so advanced notebook users can access their values.
        panel.rf_controls = dict(trace=trace_control, frequency=frequency_control,
            magnitude=magnitude_control, automatic=auto_control,
            components=dict(zip((e.name for e in self.elements), sliders)),
            transmission_lines=line_controls,
            marker1=marker_controls[0], marker2=marker_controls[1],
            level_reference=level_reference, level_button=level_button,
            reset=reset_button, status=status)
        panel.rf_state = state
        display(panel)
        Update(simulate=True)
        if state["error"] is not None:
            raise state["error"]
        return panel

    def Plot_Time(self, trace="S21", *, ax=None, xlim=None, ylim=None, show=True):
        f,s=self._result()
        match=re.fullmatch(r"S(\d+)(\d+)",trace.upper())
        if not match:
            raise ValueError("Expected Sij trace")
        i,j=int(match[1])-1,int(match[2])-1
        if not (0 <= i < len(self.ports) and 0 <= j < len(self.ports)):
            raise ValueError("Unknown trace")
        t,h=bandlimited_impulse(f,s[:,i,j])
        if ax is None:
            _,ax=plt.subplots(figsize=(9,4))
        ax.plot(t*1e9,abs(h))
        ax.set(xlabel="Time (ns)",ylabel="Normalized envelope",
               title=f"{trace.upper()} band-limited impulse · {f[0]/1e6:g}–{f[-1]/1e6:g} MHz")
        ax.grid(alpha=.3)
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if show: self.Show()
        return ax

    def Export_Touchstone(self, filename):
        f,s=self._result()
        export_touchstone(filename,f,s,self.z0)

    def Plot_Circuit(self, *, figsize=(18,7), xlim=None, ylim=None, show=True,
                     font_size=12, node_font_size=10, title_font_size=18):
        """Draw a spaced circuit with bold component and node labels."""
        self.circuit_font_size = font_size
        self.circuit_node_font_size = node_font_size
        self.figure=plt.figure(figsize=figsize)
        self.Draw_Circuit()
        ax = self.figure.axes[0]
        ax.set_title(self.title.strip() or "Untitled circuit", fontsize=title_font_size,
                     fontweight="bold", pad=26)
        for text in ax.texts:
            text.set_fontweight("bold")
            text.set_fontsize(max(text.get_fontsize(), node_font_size))
        self.figure.tight_layout(pad=2)
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if show: self.Show()
        if not show:
            return ax

    def Draw_Qucs_Schematic(self, ax, feed_line=None):
        """Draw the balanced circuit with separate lanes for labels and branches."""
        items = {e.name: e for e in self.elements}
        ink = "black"
        fs = getattr(self, "circuit_font_size", 12)
        ns = getattr(self, "circuit_node_font_size", 10)

        def wire(x1,y1,x2,y2):
            ax.plot((x1,x2),(y1,y2),color=ink,lw=1.8)

        def dot(x,y):
            ax.plot(x,y,"o",color="#c24b24",ms=6,zorder=6)

        def terminal(x,y):
            ax.plot(x,y,"o",color=ink,ms=4,zorder=5)

        def label(x,y,text,ha="center",va="center",size=None):
            ax.text(x,y,text,ha=ha,va=va,fontsize=fs if size is None else size,
                    fontweight="bold",color=ink,zorder=7)

        def component_label(name,x,y,ha="center",va="center"):
            label(x,y,f"{name}\n{component_value(items[name])}",ha,va)

        def ground(x,y):
            for dy,width in ((0,.25),(-.09,.17),(-.18,.08)):
                wire(x-width,y+dy,x+width,y+dy)
            label(x,y-.32,GROUND,va="top",size=ns)

        def cap(x,y,horizontal,name,lx,ly,ha="center",va="center"):
            if horizontal:
                wire(x-.46,y,x-.09,y);wire(x+.09,y,x+.46,y)
                wire(x-.09,y-.23,x-.09,y+.23)
                wire(x+.09,y-.23,x+.09,y+.23)
                terminal(x-.46,y);terminal(x+.46,y)
            else:
                wire(x,y+.43,x,y+.09);wire(x,y-.09,x,y-.43)
                wire(x-.23,y+.09,x+.23,y+.09)
                wire(x-.23,y-.09,x+.23,y-.09)
                terminal(x,y+.43);terminal(x,y-.43)
            component_label(name,lx,ly,ha,va)

        def coil(x,y,horizontal,name,lx,ly,ha="center",va="center"):
            t=np.linspace(-1,1,100)
            if horizontal:
                wire(x-.56,y,x-.40,y);wire(x+.40,y,x+.56,y)
                ax.plot(x+.40*t,y+.16*np.sin(5*np.pi*(t+1)),color=ink,lw=1.8)
                terminal(x-.56,y);terminal(x+.56,y)
            else:
                wire(x,y+.56,x,y+.42);wire(x,y-.42,x,y-.56)
                ax.plot(x+.16*np.sin(5*np.pi*(t+1)),y+.42*t,color=ink,lw=1.8)
                terminal(x,y+.56);terminal(x,y-.56)
            component_label(name,lx,ly,ha,va)

        def port(x,y,index,side=False):
            ax.add_patch(Circle((x,y),.18,facecolor="white",edgecolor=ink,lw=1.8,zorder=4))
            label(x,y,"P",size=ns)
            label(x-.36 if side else x,y if side else y-.40,
                  self.port_labels[index],ha="right" if side else "center",
                  va="center" if side else "top",size=ns)

        # Horizontal signal rail, with space between the two shunt branches.
        if feed_line is None:
            port(.60,0,0);wire(.78,0,1.4,0)
        else:
            port(-3.40,0,0);wire(-3.22,0,-1.80,0)
            _draw_coax_symbol(ax,-1.10,0,1,0,.70,color=ink,font_size=ns)
            terminal(-1.80,0);terminal(-.40,0);wire(-.40,0,1.40,0)
            component_label(feed_line.name,-1.10,.70,va="bottom")
        wire(1.4,0,1.94,0);cap(2.40,0,True,"C1_HM",2.40,.70,va="bottom")
        wire(2.86,0,4.20,0);wire(4.20,0,5.90,0);wire(5.90,0,6.64,0)
        cap(7.10,0,True,"C_Balance",7.10,.70,va="bottom")
        wire(7.56,0,8.0,0);wire(8.0,0,8.44,0)
        coil(9.0,0,True,"L_Sample",9.0,.70,va="bottom")
        wire(9.56,0,10.60,0);wire(10.60,0,11.10,0)
        # Parallel trap uses a higher and lower horizontal lane.
        wire(11.10,0,11.10,.60);wire(11.10,.60,11.64,.60)
        coil(12.20,.60,True,"LH_trap",12.20,1.22,va="bottom")
        wire(12.76,.60,13.30,.60)
        wire(11.10,0,11.10,-.60);wire(11.10,-.60,11.74,-.60)
        cap(12.20,-.60,True,"CH_trap",12.20,-1.12,va="top")
        wire(12.66,-.60,13.30,-.60);wire(13.30,-.60,13.30,.60)
        wire(13.30,0,14.50,0);wire(14.50,0,15.54,0)
        cap(16.0,0,True,"C1",16.0,.70,va="bottom")
        wire(16.46,0,17.20,0);wire(17.20,0,17.62,0);port(17.80,0,1)
        # Shared ports and their grounded reference terminals.
        for x,index in ((1.10 if feed_line is None else -3.00,3),(17.20,2)):
            dot(x,0);wire(x,0,x,-2.22);port(x,-2.40,index,side=True)
            wire(x,-2.58,x,-3.10);ground(x,-3.10)
        # Shunt capacitors: labels sit beside the symbols, away from wires.
        for x,name,lx,ha in ((4.20,"C2_HT",3.55,"right"),
                            (10.60,"H_Balance",9.95,"right"),
                            (14.50,"C2",15.15,"left")):
            dot(x,0);wire(x,0,x,-1.27)
            cap(x,-1.70,False,name,lx,-1.70,ha=ha)
            wire(x,-2.13,x,-3.10);ground(x,-3.10)
        dot(5.90,0);wire(5.90,0,5.90,-1.07)
        cap(5.90,-1.50,False,"C_trap1",6.55,-1.50,ha="left")
        wire(5.90,-1.93,5.90,-2.54)
        coil(5.90,-3.10,False,"L_trap1",6.55,-3.10,ha="left")
        wire(5.90,-3.66,5.90,-4.15);ground(5.90,-4.15)
        for x in (1.4,8.0,10.60,11.10,13.30,14.50,17.20):
            dot(x,0)
        core_left = (self.ports[0] if feed_line is None else
                     feed_line.node_b if feed_line.node_a == self.ports[0] else feed_line.node_a)
        if feed_line is not None:
            label(-3.00,.24,self.ports[0],va="bottom",size=ns)
        for node,x in ((core_left,1.40),("J1",5.1),("J3",8.0),
                       ("J4",10.60),("J5",14.50),(self.ports[1],17.20)):
            label(x,.24,node,va="bottom",size=ns)
        label(5.55,-2.25,"J2",ha="right",size=ns)
        ax.set(xlim=(-.15 if feed_line is None else -4.50,18.5),
               ylim=(-4.9,2.1),aspect="equal")
        ax.axis("off")

    def Draw_Circuit(self):
        ax = self.figure.add_subplot(111)
        left_node = self.ports[0]
        right_node = self.ports[1] if len(self.ports) > 1 else None
        feed_lines = [e for e in self.elements if isinstance(e,TransmissionLine)
                      and left_node in (e.node_a,e.node_b) and GROUND not in (e.node_a,e.node_b)]
        feed_line = feed_lines[0] if len(feed_lines) == 1 else None
        core_left = (left_node if feed_line is None else
                     feed_line.node_b if feed_line.node_a == left_node else feed_line.node_a)
        expected={
            "C1_HM":(core_left,"J1"), "C2_HT":("J1",GROUND),
            "C_trap1":("J1","J2"), "L_trap1":("J2",GROUND),
            "C_Balance":("J1","J3"), "L_Sample":("J3","J4"),
            "H_Balance":("J4",GROUND), "LH_trap":("J4","J5"),
            "CH_trap":("J4","J5"), "C2":("J5",GROUND),
            "C1":("J5",right_node)}
        actual={e.name: {e.node_a,e.node_b} for e in self.elements if e is not feed_line}
        kinds={e.name:e.kind for e in self.elements}
        inductors={"L_trap1","L_Sample","LH_trap"}
        if (len(self.ports)==4 and self.ports==[left_node,right_node,right_node,left_node] and
            len(actual)==len(expected) and
            all(actual.get(name)==set(nodes) and
                kinds[name]==("L" if name in inductors else "C")
                for name,nodes in expected.items())):
            self.Draw_Qucs_Schematic(ax,feed_line=feed_line)
            ax.set_title(self.circuit_title.get().strip() or "Untitled circuit",pad=24)
            return
        self.Draw_Spaced_Schematic(ax)

    def Draw_Spaced_Schematic(self, ax):
        """Use a verified topology layout or separated, electrically labeled branches."""
        import textwrap
        fs = getattr(self, "circuit_font_size", 12)
        ns = getattr(self, "circuit_node_font_size", 10)
        size_scale = max(1, fs/12, ns/10)
        ink = "black"
        items = {e.name:e for e in self.elements}
        ax.set_aspect("equal")
        ax.axis("off")

        def wire(a, b):
            ax.plot([a[0],b[0]], [a[1],b[1]], color=ink, lw=1.6)

        def ground(x, y):
            for offset,width in ((0,.28),(-.10,.19),(-.20,.09)):
                wire((x-width,y+offset),(x+width,y+offset))
            ax.text(x,y-.38,GROUND,ha="center",va="top",fontsize=ns)

        def part(e, a, b, side=1):
            x1,y1=a; x2,y2=b
            dx,dy=x2-x1,y2-y1
            length=np.hypot(dx,dy)
            if length <= 0 or (abs(dx)>1e-9 and abs(dy)>1e-9):
                raise ValueError("Schematic components require a nonzero orthogonal segment")
            ux,uy=dx/length,dy/length
            nx,ny=-uy,ux
            mx,my=(x1+x2)/2,(y1+y2)/2
            half=min(.5,length*.25)
            left=(mx-ux*half,my-uy*half)
            right=(mx+ux*half,my+uy*half)
            wire(a,left);wire(right,b)
            if e.kind=="C":
                gap=.09
                for offset in (-gap,gap):
                    cx,cy=mx+ux*offset,my+uy*offset
                    wire((cx-nx*.25,cy-ny*.25),(cx+nx*.25,cy+ny*.25))
                wire(left,(mx-ux*gap,my-uy*gap))
                wire((mx+ux*gap,my+uy*gap),right)
            elif e.kind=="TL":
                _draw_coax_symbol(ax,mx,my,ux,uy,half,font_size=ns)
            else:
                t=np.linspace(-1,1,81) if e.kind=="L" else np.linspace(-1,1,11)
                v=(.18*np.sin(4*np.pi*(t+1)) if e.kind=="L" else
                   np.array([0,.18,-.18,.18,-.18,.18,-.18,.18,-.18,.18,0]))
                ax.plot(mx+ux*half*t+nx*v,my+uy*half*t+ny*v,color=ink,lw=1.6)
            for point in (left,right):
                ax.plot(*point,"o",color=ink,ms=3.7,zorder=5)
            offset=1.15 if e.kind=="TL" else .8
            tx,ty=mx+side*nx*offset,my+side*ny*offset
            horizontal=abs(dx)>abs(dy)
            ax.text(tx,ty,textwrap.fill(e.name, width=18)+f"\n{component_value(e)}",fontsize=fs,
                    ha="center" if horizontal else ("left" if tx>mx else "right"),
                    va="center",bbox=dict(facecolor="white",edgecolor="none",pad=1))

        expected={
            "L1":("J1","J2"),"L2":("J1","J3"),"L3":("J2","J4"),
            "L4":("J3","J5"),"L5":("J4","J6"),"L6":("N2","J7"),
            "C1":("J1","J2"),"C2":("J3",GROUND),"C3":("J4","N1"),
            "C4":("N1",GROUND),"C5":("J6",GROUND),"C6":("J5",GROUND),
            "C7":("J7","J5"),"C8":("N2","J7"),"C9":("J7",GROUND),
            "C10":("N1",GROUND),"C11":("N2",GROUND),"C12":("J5",GROUND),
            "C13":("J6",GROUND),"C14":("J3",GROUND),"C15":("J4",GROUND),
            "R1":("J1",GROUND),"R2":("J2",GROUND)}
        is_b1=(len(items)==len(self.elements)==len(expected) and
               all(name in items and items[name].kind==name[0] and
                   {items[name].node_a,items[name].node_b}==set(nodes)
                   for name,nodes in expected.items()) and self.ports==["N1","N2"])
        if is_b1:
            width,height=self.figure.get_size_inches()
            self.figure.set_size_inches(max(width,20*size_scale),max(height,10*size_scale))
            # Reserve separate lanes for each shunt, series part and its label.
            pos={"J1":(0,12),"J2":(14,12),"J3":(0,6),"J4":(14,6),
                 "J5":(0,0),"J6":(14,0),"J7":(-7,0),"N2":(-14,0),"N1":(21,6)}
            for name in ("L1","L2","L3","L4","L5","C3","C7"):
                a,b=expected[name];part(items[name],pos[a],pos[b])
            for name,y in (("C1",10.5),("L6",0),("C8",1.8)):
                a,b=expected[name];xa,ya=pos[a];xb,yb=pos[b]
                wire((xa,ya),(xa,y));wire((xb,y),(xb,yb))
                part(items[name],(xa,y),(xb,y),side=1 if name!="L6" else -1)
            for name,x,side in (("C2",-3,-1),("C14",3,1),("C15",11,-1),
                               ("C4",20,-1),("C10",24,1),
                               ("C6",-2.5,-1),("C12",2.5,1),
                               ("C5",11.5,-1),("C13",16.5,1),
                               ("C9",-7,-1),("C11",-14,-1)):
                node=expected[name][0];sx,sy=pos[node]
                wire((sx,sy),(x,sy));part(items[name],(x,sy),(x,sy-3.4),side=side)
                ground(x,sy-3.4)
            for name,x in (("R1",-5),("R2",19)):
                node=expected[name][0];start=pos[node];end=(x,12)
                part(items[name],start,end,side=1 if x>start[0] else -1)
                wire(end,(x,11.3));ground(x,11.3)
            for node,(x,y) in pos.items():
                ax.plot(x,y,"o",color="#c24b24",ms=6,zorder=6)
                ax.text(x+.18,y+.25,node,fontsize=ns,ha="left",va="bottom")
            for label,node in self.port_definitions:
                x,y=pos[node]
                # Place ports above the feed so they cannot cover a component.
                wire((x,y),(x,y+2.7))
                ax.add_patch(Circle((x,y+3),.3,facecolor="white",edgecolor=ink,lw=1.6))
                ax.text(x,y+3,label,ha="center",va="center",fontsize=ns)
            ax.set_xlim(-18,28);ax.set_ylim(-5.3,14.5)
            return

        # General fallback: separate branch cells, connected by named nets.
        # Equal node labels denote the same electrical node throughout the sheet.
        # This avoids ambiguous wire crossings and shared drawing lanes.
        count=len(self.elements)
        columns=min(3,max(1,count))
        rows=(count+columns-1)//columns
        cell_width,cell_height=10.5,5.5
        for i,e in enumerate(self.elements):
            column,row=i%columns,i//columns
            cx=column*cell_width;cy=-row*cell_height
            part(e,(cx-2.5,cy),(cx+2.5,cy))
            for x,node in ((cx-2.5,e.node_a),(cx+2.5,e.node_b)):
                ax.plot(x,cy,"o",color="#c24b24" if node!=GROUND else ink,ms=5,zorder=6)
                if node==GROUND:
                    wire((x,cy),(x,cy-.55));ground(x,cy-.55)
                else:
                    labels=[label for label,pnode in self.port_definitions if pnode==node]
                    text=node+(" / "+", ".join(labels) if labels else "")
                    ax.text(x,cy-.45,textwrap.fill(text,width=16),ha="center",va="top",fontsize=ns)
        footer_y=-(rows-1)*cell_height-2.5
        ax.text((columns-1)*cell_width/2,footer_y,
                "Equal node labels are electrically connected; P0 is ground.",
                ha="center",va="top",fontsize=ns)
        ax.set_xlim(-4.5,(columns-1)*cell_width+4.5)
        ax.set_ylim(footer_y-1,2.5)
        # Grow the figure with the number of components to preserve text space.
        width,height=self.figure.get_size_inches()
        self.figure.set_size_inches(max(width,columns*6*size_scale),max(height,(rows*2.8+1.5)*size_scale))
