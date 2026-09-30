"""
PyOR Python On Resonance

Author: Vineeth Francis Thalakottoor Jose Chacko

Email: vineethfrancis.physics@gmail.com

Description:
    This file contains the classes `RFCircuit` and `Element`.
"""

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

__all__ = ["Element", "RFCircuit"]


PREFIX = {"": 1, "k": 1e3, "M": 1e6, "G": 1e9,
          "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}
GROUND = "P0"
UNIT_SCALE = {"Ω": 1, "kΩ": 1e3, "MΩ": 1e6,
              "pH": 1e-12, "nH": 1e-9, "µH": 1e-6, "mH": 1e-3, "H": 1,
              "fF": 1e-15, "pF": 1e-12, "nF": 1e-9, "µF": 1e-6, "mF": 1e-3, "F": 1}
KIND_UNITS = {"R": ("Ω", "kΩ", "MΩ"),
              "L": ("pH", "nH", "µH", "mH", "H"),
              "C": ("fF", "pF", "nF", "µF", "mF", "F")}


def normalize_node(node):
    node = str(node).strip()
    return GROUND if node.upper() in ("0", "P0") else node


def component_value(e):
    units = KIND_UNITS[e.kind]
    if e.display_unit in units:
        unit = e.display_unit
    else:
        unit = next((u for u in reversed(units) if e.value >= UNIT_SCALE[u]), units[0])
    return f"{e.value/UNIT_SCALE[unit]:.5g} {unit}"


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

    def __init__(self, name, kind, node_a, node_b, value, unit=""):
        """Define a component with an explicit unit, e.g. value=0.3045, unit='pF'.

        With no unit, value uses SI (ohms, henries or farads). Internally .value
        always stores SI so a unit is converted exactly once.
        """
        self.name = str(name).strip()
        self.kind = str(kind).strip().upper()
        self.node_a, self.node_b = normalize_node(node_a), normalize_node(node_b)
        if self.kind not in KIND_UNITS:
            raise ValueError("Component kind must be R, L or C")
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
        if e.kind not in ("R", "L", "C") or e.node_a == e.node_b:
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
    if f.ndim != 1 or len(f) < 3 or np.any(f <= 0) or not np.all(np.isfinite(f)):
        raise ValueError("Use at least three positive frequency samples")
    if z0 <= 0 or not np.isfinite(z0):
        raise ValueError("Z0 must be positive")
    elements = [Element.From_SI(e.name,e.kind,normalize_node(e.node_a),normalize_node(e.node_b),e.value,
                        e.display_unit) for e in elements]
    port_nodes = [normalize_node(n) for n in port_nodes]
    validate_network(elements, port_nodes)
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


def smith_grid(ax, z0=50):
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
    """Ideal RLC network with a Jupyter-friendly simulation and plotting API."""

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
                elements = [Element(item["name"], item["kind"], item["node_a"],
                            item["node_b"], item["value"], item["unit"])
                            for item in data["components"]]
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
            ordinate=(np.unwrap(np.angle(value))*180/np.pi if phase else
                      20*np.log10(np.maximum(abs(value),1e-300)))
            ax.plot(f/1e6,ordinate,label=trace.upper())
        ax.set(xlabel="Frequency (MHz)",ylabel="Phase (degrees)" if phase else "Magnitude (dB)",
               title=self.title)
        ax.grid(alpha=.3);ax.legend()
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
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

    def Plot_Smith(self, port=1, *, ax=None, xlim=None, ylim=None, show=True):
        f,s=self._result()
        if not 1 <= port <= len(self.ports):
            raise ValueError("Unknown port number")
        if ax is None:
            _,ax=plt.subplots(figsize=(7,7))
        smith_grid(ax,self.z0)
        gamma=s[:,port-1,port-1]
        ax.plot(gamma.real,gamma.imag,label=f"S{port}{port}")
        best=np.argmin(abs(gamma))
        ax.plot(gamma[best].real,gamma[best].imag,"o",color="#c24b24")
        z=self.Input_Impedance(port)[best]
        ax.text(.02,.02,f"Closest match: {f[best]/1e6:.6g} MHz\n"
                     f"Zin={z.real:.4g}{z.imag:+.4g}j Ω",
                transform=ax.transAxes,fontsize=8,
                bbox=dict(facecolor="white",edgecolor=".8",pad=4))
        ax.legend(loc="upper right")
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if show: self.Show()
        return ax

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

    def Plot_Circuit(self, *, figsize=(15,5), xlim=None, ylim=None, show=True):
        self.figure=plt.figure(figsize=figsize)
        self.Draw_Circuit()
        ax = self.figure.axes[0]
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if show: self.Show()
        return self.figure.axes[0]

    def Draw_Qucs_Schematic(self, ax):
        """Draw the uploaded balanced series-trap topology and four ports."""
        needed={"C1_HM","C2_HT","C_trap1","L_trap1","C_Balance",
                "L_Sample","H_Balance","LH_trap","CH_trap","C2","C1"}
        if not needed.issubset({e.name for e in self.elements}) or len(self.ports)!=4:
            ax.axis("off")
            ax.text(.5,.5,"Four-port Qucs layout applies to the balanced series-trap project",
                    ha="center",va="center",transform=ax.transAxes)
            return
        items={e.name:e for e in self.elements}
        ink="black"
        junction_color="#c24b24"
        def wire(x1,y1,x2,y2):
            ax.plot((x1,x2),(y1,y2),color=ink,lw=1.7)
        def dot(x,y): ax.plot(x,y,"o",color=junction_color,ms=5.5,zorder=6)
        def terminal(x,y): ax.plot(x,y,"o",color=ink,ms=3.5,zorder=5)
        def ground(x,y):
            for dy,width in ((0,.20),(-.075,.13),(-.15,.06)):
                wire(x-width,y+dy,x+width,y+dy)
        def cap(x,y,horizontal,name,above=True):
            if horizontal:
                wire(x-.38,y,x-.075,y);wire(x+.075,y,x+.38,y)
                wire(x-.075,y-.17,x-.075,y+.17)
                wire(x+.075,y-.17,x+.075,y+.17)
                terminal(x-.38,y);terminal(x+.38,y)
                ax.text(x,y+.30 if above else y-.29,
                        f"{name}  {component_value(items[name])}",
                        ha="center",va="bottom" if above else "top",fontsize=7.5)
            else:
                wire(x,y+.32,x,y+.07);wire(x,y-.07,x,y-.32)
                wire(x-.17,y+.07,x+.17,y+.07)
                wire(x-.17,y-.07,x+.17,y-.07)
                terminal(x,y+.32);terminal(x,y-.32)
                if name=="H_Balance":
                    ax.text(x,-1.67,f"{name}  {component_value(items[name])}",
                            ha="center",va="top",fontsize=7)
                else:
                    left_label=name=="C2_HT"
                    ax.text(x-.22 if left_label else x+.22,y,
                            f"{name}  {component_value(items[name])}",
                            ha="right" if left_label else "left",va="center",fontsize=7)
        def coil(x,y,horizontal,name,above=True):
            t=np.linspace(-1,1,90)
            if horizontal:
                wire(x-.45,y,x-.32,y);wire(x+.32,y,x+.45,y)
                ax.plot(x+.32*t,y+.11*np.sin(5*np.pi*(t+1)),color=ink,lw=1.5)
                terminal(x-.45,y);terminal(x+.45,y)
                ax.text(x,y+(.56 if name=="L_Sample" else .27) if above else y-.26,
                        f"{name}  {component_value(items[name])}",ha="center",
                        va="bottom" if above else "top",fontsize=7.5)
            else:
                wire(x,y+.39,x,y+.31);wire(x,y-.31,x,y-.39)
                ax.plot(x+.11*np.sin(5*np.pi*(t+1)),y+.31*t,color=ink,lw=1.5)
                terminal(x,y+.39);terminal(x,y-.39)
                ax.text(x+.2,y,f"{name}  {component_value(items[name])}",
                        ha="left",va="center",fontsize=7)
        def port(x,y,label,horizontal=True):
            ax.add_patch(Circle((x,y),.13,fill=False,lw=1.5,edgecolor=ink))
            ax.text(x,y,"P",ha="center",va="center",fontsize=7,color=ink)
            ax.text(x,y-.20 if horizontal else y-.62,label,
                    ha="center",va="top",fontsize=7.3)
        # Main rail from the proton side to the carbon side.
        wire(.75,0,1.20,0);cap(1.55,0,True,"C1_HM");wire(1.93,0,3.15,0)
        wire(3.15,0,4.38,0);cap(4.75,0,True,"C_Balance")
        wire(5.13,0,5.55,0);coil(6.0,0,True,"L_Sample")
        wire(6.45,0,7.18,0);wire(7.18,0,7.55,0)
        wire(7.55,0,7.55,.39);wire(7.55,.39,7.87,.39)
        coil(8.28,.39,True,"LH_trap");wire(8.73,.39,9.23,.39)
        wire(7.55,0,7.55,-.39);wire(7.55,-.39,7.89,-.39)
        cap(8.27,-.39,True,"CH_trap",False);wire(8.65,-.39,9.23,-.39)
        wire(9.23,-.39,9.23,.39);wire(9.23,0,10.28,0)
        wire(10.28,0,10.75,0);cap(11.13,0,True,"C1")
        wire(11.51,0,12.45,0)
        # Four independently terminated Qucs ports, two at each signal node.
        port(.50,0,self.port_labels[0]);wire(.63,0,.75,0)
        port(12.70,0,self.port_labels[1]);wire(12.45,0,12.57,0)
        for px,label in ((.92,self.port_labels[3]),(12.05,self.port_labels[2])):
            dot(px,0);wire(px,0,px,-.75);port(px,-.88,label,False)
            wire(px,-1.01,px,-1.14);ground(px,-1.14)
        # Shunt matching branches and the series trap to ground.
        for bx,cy,name in ((3.15,-.61,"C2_HT"),(7.18,-.61,"H_Balance"),
                            (10.28,-.61,"C2")):
            dot(bx,0);wire(bx,0,bx,cy+.32);cap(bx,cy,False,name)
            wire(bx,cy-.32,bx,-1.37);ground(bx,-1.37)
        dot(4.02,0);wire(4.02,0,4.02,-.28)
        cap(4.02,-.60,False,"C_trap1")
        wire(4.02,-.92,4.02,-1.05)
        coil(4.02,-1.44,False,"L_trap1")
        wire(4.02,-1.83,4.02,-1.98);ground(4.02,-1.98)
        for xnode in (1.15,3.15,4.02,5.55,7.18,7.55,9.23,10.28,12.05):
            dot(xnode,0)
        for node,x,y in ((self.ports[0],1.15,.16),("J1",3.65,.16),
                         ("J2",4.02,-1.0),("J3",5.55,.16),
                         ("J4",7.18,.16),("J5",10.28,.16),(self.ports[1],12.05,.16)):
            ax.text(x,y,node,ha="center" if node!="J2" else "right",
                    va="bottom",fontsize=7,color=ink,
                    bbox=dict(facecolor="white",edgecolor="none",pad=.6))
        for x,y in ((.92,-1.32),(3.15,-1.55),(4.02,-2.16),
                    (7.18,-1.55),(10.28,-1.55),(12.05,-1.32)):
            ax.text(x,y,GROUND,ha="center",va="top",fontsize=7,color=ink)
        ax.set(xlim=(.1,13.2),ylim=(-2.25,.88),aspect="equal")
        ax.axis("off")

    def Draw_Circuit(self):
        ax = self.figure.add_subplot(111)
        left_node = self.ports[0]
        right_node = self.ports[1] if len(self.ports) > 1 else None
        expected={
            "C1_HM":(left_node,"J1"), "C2_HT":("J1",GROUND),
            "C_trap1":("J1","J2"), "L_trap1":("J2",GROUND),
            "C_Balance":("J1","J3"), "L_Sample":("J3","J4"),
            "H_Balance":("J4",GROUND), "LH_trap":("J4","J5"),
            "CH_trap":("J4","J5"), "C2":("J5",GROUND),
            "C1":("J5",right_node)}
        actual={e.name: {e.node_a,e.node_b} for e in self.elements}
        kinds={e.name:e.kind for e in self.elements}
        inductors={"L_trap1","L_Sample","LH_trap"}
        if (len(self.ports)==4 and self.ports==[left_node,right_node,right_node,left_node] and
            len(actual)==len(expected) and
            all(actual.get(name)==set(nodes) and
                kinds[name]==("L" if name in inductors else "C")
                for name,nodes in expected.items())):
            self.Draw_Qucs_Schematic(ax)
            ax.set_title(self.circuit_title.get().strip() or "Untitled circuit",pad=24)
            return
        ax.set(title=self.circuit_title.get().strip() or "Untitled circuit", aspect="equal")
        ax.axis("off")
        color = "black"
        junction_color = "#c24b24"
        internal = sorted({v for e in self.elements for v in (e.node_a,e.node_b)
                           if v != GROUND and v not in self.ports})
        if len(self.ports) == 1:
            # Put a one-port ladder on a horizontal signal rail. Each shunt
            # component gets its own vertical branch beneath its node.
            positions = {self.ports[0]: (0.0, 2.5)}
            queue = [self.ports[0]]
            while queue:
                node = queue.pop(0)
                neighbors = sorted({other for e in self.elements
                                    for other in ((e.node_b,) if e.node_a == node else
                                                  (e.node_a,) if e.node_b == node else ())
                                    if other != GROUND and other not in positions})
                for other in neighbors:
                    positions[other] = (5.2 * len(positions), 2.5)
                    queue.append(other)
            for other in internal:
                if other not in positions:
                    positions[other] = (5.2 * len(positions), 2.5)
        else:
            unique_ports = list(dict.fromkeys(self.ports))
            positions = {node: (i*5.2-(len(unique_ports)-1)*2.6, 3.2)
                         for i,node in enumerate(unique_ports)}
            for i,node in enumerate(internal):
                positions[node] = (i*5.2-(len(internal)-1)*2.6, -1.3)
            # A three-way splitter reads as a T: side ports on the junction
            # rail and its middle port above the junction.
            if len(unique_ports)==3 and len(internal)==1:
                positions[unique_ports[0]]=(positions[unique_ports[0]][0],-1.3)
                positions[unique_ports[2]]=(positions[unique_ports[2]][0],-1.3)
        positions.update({node:tuple(self.layout_positions[node]) for node in positions
                          if node in self.layout_positions})

        def component(e, p1, p2, label_side=1):
            x1,y1 = p1; x2,y2 = p2
            dx,dy = x2-x1,y2-y1
            length = np.hypot(dx,dy)
            if length < .1:
                return
            ux,uy = dx/length,dy/length
            nx,ny = -uy,ux
            mx,my = (x1+x2)/2,(y1+y2)/2
            half = min(.43, length*.24)
            left = (mx-ux*half,my-uy*half)
            right = (mx+ux*half,my+uy*half)
            ax.plot([x1,left[0]],[y1,left[1]],color=color,lw=1.6)
            ax.plot([right[0],x2],[right[1],y2],color=color,lw=1.6)
            if e.kind == "C":
                gap = min(.09,half*.3)
                for offset in (-gap,gap):
                    cx,cy = mx+ux*offset,my+uy*offset
                    ax.plot([cx-nx*.24,cx+nx*.24],[cy-ny*.24,cy+ny*.24],
                            color=color,lw=2.2)
                ax.plot([left[0],mx-ux*gap],[left[1],my-uy*gap],color=color,lw=1.6)
                ax.plot([mx+ux*gap,right[0]],[my+uy*gap,right[1]],color=color,lw=1.6)
            elif e.kind == "R":
                # Conventional zigzag resistor, oriented along its wire.
                steps = [(-1,0),(-.8,.18),(-.6,-.18),(-.4,.18),
                         (-.2,-.18),(0,.18),(.2,-.18),(.4,.18),
                         (.6,-.18),(.8,.18),(1,0)]
                ax.plot([mx+ux*half*t+nx*v for t,v in steps],
                        [my+uy*half*t+ny*v for t,v in steps],color=color,lw=1.6)
            else:
                t=np.linspace(-1,1,80)
                coil=np.sin(4*np.pi*(t+1))* .17
                ax.plot(mx+ux*half*t+nx*coil,my+uy*half*t+ny*coil,color=color,lw=1.6)
            # Black dots mark the two wire-to-component terminals.
            ax.plot(left[0],left[1],"o",color=color,ms=3.7,zorder=5)
            ax.plot(right[0],right[1],"o",color=color,ms=3.7,zorder=5)
            side = label_side
            label_x,label_y=mx+nx*.6*side,my+ny*.6*side
            horizontal=abs(dx)>abs(dy)
            ax.text(label_x,label_y,f"{e.name} = {component_value(e)}",
                    ha="center" if horizontal else ("left" if label_x>mx else "right"),
                    va="center",fontsize=8,
                    bbox=dict(facecolor="white",edgecolor="none",pad=1))

        # Route every connection on horizontal and vertical tracks. Parallel
        # elements get separate tracks while each symbol stays on one axis.
        groups = {}
        for e in self.elements:
            if GROUND not in (e.node_a,e.node_b):
                groups.setdefault(tuple(sorted((e.node_a,e.node_b))),[]).append(e)
        for (a,b),group in groups.items():
            xa,ya=positions[a]; xb,yb=positions[b]
            dx,dy=xb-xa,yb-ya
            horizontal=abs(dx)>=abs(dy)
            for i,e in enumerate(group):
                offset=(i-(len(group)-1)/2)*1.55
                if horizontal:
                    lane=(ya+yb)/2+offset
                    pa,pb=(xa,lane),(xb,lane)
                    ax.plot([xa,xa],[ya,lane],color=color,lw=1.6)
                    ax.plot([xb,xb],[lane,yb],color=color,lw=1.6)
                else:
                    lane=(xa+xb)/2+offset
                    pa,pb=(lane,ya),(lane,yb)
                    ax.plot([xa,lane],[ya,ya],color=color,lw=1.6)
                    ax.plot([lane,xb],[yb,yb],color=color,lw=1.6)
                # Put labels outside parallel tracks, away from their symbols.
                normal_sign = (1 if xb > xa else -1) if horizontal else (-1 if yb > ya else 1)
                outward = 1 if offset >= 0 else -1
                component(e,pa,pb,label_side=outward*normal_sign)

        # Ground is a common electrical node, but shunt parts have distinct
        # ground symbols at their own x coordinates in a conventional drawing.
        grounded = {}
        for e in self.elements:
            if GROUND in (e.node_a,e.node_b):
                node = e.node_b if e.node_a == GROUND else e.node_a
                grounded.setdefault(node,[]).append(e)
        ground_x = []
        for node,group in grounded.items():
            x,y = positions[node]
            for i,e in enumerate(group):
                branch_x = x + (i-(len(group)-1)/2)*1.65
                if branch_x != x:
                    ax.plot([x,branch_x],[y,y],color=color,lw=1.6)
                gy = y-3.2
                component(e,(branch_x,y),(branch_x,gy),label_side=-1)
                for offset,width in ((0,.32),(-.09,.22),(-.18,.11)):
                    ax.plot([branch_x-width,branch_x+width],[gy+offset,gy+offset],
                            color=color,lw=1.4)
                ground_x.append(branch_x)
                ax.text(branch_x,gy-.35,GROUND,ha="center",va="top",fontsize=8)

        for node in dict.fromkeys(self.ports):
            x,y = positions[node]
            labels = ", ".join(self.port_labels[i] for i,p in enumerate(self.ports) if p == node)
            ax.add_patch(Circle((x,y),.28,facecolor="white",
                                edgecolor=color,lw=1.6,zorder=3))
            ax.text(x,y,"P",ha="center",va="center",fontsize=9,zorder=4)
            ax.text(x,y+.39,f"{labels} · {node}",ha="center",va="bottom",fontsize=8,
                    bbox=dict(facecolor="white",edgecolor="none",pad=1),zorder=4)
        for node in internal:
            x,y=positions[node]
            ax.plot(x,y,"o",color=junction_color,ms=8,zorder=6)
            ax.text(x+.2,y+.2,node,ha="left",va="bottom",fontsize=9)
        all_x=[v[0] for v in positions.values()]+ground_x
        all_y=[v[1] for v in positions.values()]
        ax.set(xlim=(min(all_x)-2.0,max(all_x)+2.0),
               ylim=(min(all_y)-4.2 if grounded else min(all_y)-1.5,
                     max(all_y)+1.5))
        self.figure.tight_layout()
