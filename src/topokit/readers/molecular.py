"""Dependency-free molecular and crystallographic point readers.

The adapters in this module extract atom sites and format metadata only.  They
do not infer topology, assign feature weights, expand crystal symmetry, or
choose a downstream construction.
"""

from __future__ import annotations

import math
import re
import shlex
from pathlib import Path

from ..data import PointCloud
from .atomic_properties import PAULING_ELECTRONEGATIVITY
from .formats import _units


_ELEMENTS = frozenset(PAULING_ELECTRONEGATIVITY)
_CIF_NUMBER = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)(?:\(\d+\))?$"
)
_SDF_FIELD = re.compile(r"^>\s*<([^>]+)>")


class _CifToken(str):
    """A CIF token retaining whether delimiters made it a quoted value."""

    def __new__(cls, value, *, quoted=False):
        token = super().__new__(cls, value)
        token.quoted = quoted
        return token


def _read_lines(path):
    return Path(path).read_text(encoding="utf-8-sig").splitlines()


def _id(token):
    token = token.strip()
    try:
        return int(token)
    except ValueError:
        return token


def _float(token, context):
    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite")
    return value


def _integer(token, context):
    try:
        return int(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer") from exc


def _optional_float(token, context):
    token = token.strip()
    return None if not token else _float(token, context)


def _canonical_element(value):
    """Return a canonical element token when a format encodes one."""
    value = value.strip()
    if value in _ELEMENTS:
        return value
    if value.title() in _ELEMENTS:
        return value.title()
    letters = re.match(r"[A-Za-z]+", value)
    if letters:
        stem = letters.group(0)
        for width in (2, 1):
            candidate = stem[:width].title()
            if candidate in _ELEMENTS:
                return candidate
    return value


def _cif_key(name):
    """Normalize core-CIF underscore and mmCIF dotted data names."""
    return name.lower().replace(".", "_")


def _pdb_element(atom_name, element_field=""):
    if element_field.strip():
        return _canonical_element(element_field)
    # PDB atom-name alignment distinguishes protein `` CA `` (carbon) from
    # ``CA  `` (calcium).  This fallback is used only when columns 77-78 are
    # absent and the original atom name is retained alongside it.
    if atom_name.startswith(" "):
        match = re.search(r"[A-Za-z]", atom_name)
        return _canonical_element(match.group(0)) if match else atom_name.strip()
    return _canonical_element(atom_name.strip())


def _pdbqt_element(atom_type, atom_name):
    token = atom_type.strip()
    aliases = {
        "A": "C", "NA": "N", "NS": "N", "OA": "O", "OS": "O",
        "SA": "S", "HD": "H", "HS": "H",
    }
    return aliases.get(token.upper(), _canonical_element(token or atom_name.strip()))


def _point_cloud(path, format_name, points, ids, labels, columns, coordinate_units,
                 **format_metadata):
    point_ids = tuple(ids)
    labels = tuple(labels)
    columns = {name: tuple(values) for name, values in columns.items()}
    point_count = len(point_ids)
    if len(labels) != point_count or any(len(values) != point_count for values in columns.values()):
        raise ValueError(f"{format_name.upper()} atom attributes are not aligned")
    metadata = {
        "source_file": Path(path).name,
        "format": format_name,
        "coordinate_units": coordinate_units,
        "labels": labels,
        "columns": columns,
        "atom_count": point_count,
        "source_atom_count": point_count,
        "selected_atom_count": point_count,
        "selection_source_ids": point_ids,
        **format_metadata,
    }
    if "bonds" in metadata:
        metadata["source_bonds"] = tuple(metadata["bonds"])
        metadata["source_bond_count"] = len(metadata["source_bonds"])
        metadata["selected_bond_count"] = len(metadata["bonds"])
        metadata["bond_selection_policy"] = "source_record"
    if "connectivity_records" in metadata:
        metadata["source_connectivity_records"] = tuple(
            metadata.get("source_connectivity_records", metadata["connectivity_records"])
        )
        metadata["source_connectivity_record_count"] = len(metadata["source_connectivity_records"])
        metadata["selected_connectivity_record_count"] = len(metadata["connectivity_records"])
        metadata.setdefault("connectivity_selection_policy", "source_record")
    if "declared_counts" in metadata:
        metadata["source_declared_counts"] = dict(metadata["declared_counts"])
    try:
        return PointCloud(points, ids=point_ids, metadata=metadata)
    except ValueError as exc:
        if "point IDs must be unique" in str(exc):
            raise ValueError(f"{format_name.upper()} atom IDs must be unique") from exc
        raise


def _selected_pdb_atoms(lines, model):
    model_markers = []
    atom_records = []
    active_model = None
    for line in lines:
        record = line[:6].strip().upper()
        if record == "MODEL":
            marker = line[10:14].strip() or line[6:].strip()
            active_model = _id(marker) if marker else len(model_markers) + 1
            model_markers.append(active_model)
        elif record == "ENDMDL":
            active_model = None
        elif record in {"ATOM", "HETATM"}:
            atom_records.append((active_model, line))
    if not atom_records:
        raise ValueError("PDB-family file contains no ATOM or HETATM records")
    available = tuple(dict.fromkeys(marker for marker, _ in atom_records if marker is not None))
    if not available:
        if model not in (None, 1, "1"):
            raise ValueError("model was requested but the file has no MODEL records")
        return tuple(line for _, line in atom_records), None, ()
    if model is None:
        if len(available) != 1:
            raise ValueError("multiple coordinate models require an explicit model= selection")
        selected = available[0]
    else:
        selected = _id(str(model))
        if selected not in available:
            raise ValueError(f"requested model {model!r} is not present")
    return tuple(line for marker, line in atom_records if marker == selected), selected, available


def _pdb_formal_charge(token):
    token = token.strip()
    if not token:
        return None
    if token == "0":
        return 0
    if re.fullmatch(r"\d+[+-]", token):
        magnitude = int(token[:-1])
        return magnitude if token[-1] == "+" else -magnitude
    if re.fullmatch(r"[+-]\d+", token):
        return int(token)
    raise ValueError(f"invalid PDB formal charge {token!r}")


def _pdb_cell(lines):
    record = next((line for line in lines if line.startswith("CRYST1")), None)
    if record is None:
        return None
    if len(record) < 54:
        raise ValueError("PDB CRYST1 record is truncated")
    return {
        "length_a": _float(record[6:15], "PDB cell length a"),
        "length_b": _float(record[15:24], "PDB cell length b"),
        "length_c": _float(record[24:33], "PDB cell length c"),
        "angle_alpha": _float(record[33:40], "PDB cell angle alpha"),
        "angle_beta": _float(record[40:47], "PDB cell angle beta"),
        "angle_gamma": _float(record[47:54], "PDB cell angle gamma"),
        "space_group": record[55:66].strip() or None,
        "z": _id(record[66:70]) if record[66:70].strip() else None,
    }


def _pdb_connectivity(lines):
    records = []
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        fields = [line[index:index + 5].strip() for index in range(6, len(line), 5)]
        fields = tuple(_id(field) for field in fields if field)
        if fields:
            records.append(fields)
    return tuple(records)


def _induced_connectivity(records, selected_ids):
    """Restrict source CONECT records to a selected coordinate-model view."""
    selected = set(selected_ids)
    induced = []
    for record in records:
        if not record or record[0] not in selected:
            continue
        retained = (record[0],) + tuple(atom_id for atom_id in record[1:]
                                        if atom_id in selected)
        if len(retained) >= 2:
            induced.append(retained)
    return tuple(induced)


def read_pdb(path, *, coordinate_units="angstrom", model=None):
    """Read one PDB coordinate model without inferring chemical bonds."""
    coordinate_units = _units(coordinate_units)
    lines = _read_lines(path)
    atom_lines, selected_model, available_models = _selected_pdb_atoms(lines, model)
    rows = []
    for line in atom_lines:
        if len(line) < 54:
            raise ValueError("PDB atom record is truncated")
        serial = line[6:11].strip()
        if not serial:
            raise ValueError("PDB atom record is missing its serial ID")
        atom_name = line[12:16]
        rows.append({
            "record_name": line[:6].strip().upper(),
            "serial": _id(serial),
            "atom_name": atom_name.strip(),
            "alternate_location": line[16:17].strip() or None,
            "residue_name": line[17:20].strip() or None,
            "chain_id": line[21:22].strip() or None,
            "residue_sequence": _id(line[22:26]) if line[22:26].strip() else None,
            "insertion_code": line[26:27].strip() or None,
            "x": _float(line[30:38], "PDB x coordinate"),
            "y": _float(line[38:46], "PDB y coordinate"),
            "z": _float(line[46:54], "PDB z coordinate"),
            "occupancy": _optional_float(line[54:60], "PDB occupancy") if len(line) >= 60 else None,
            "temperature_factor": (
                _optional_float(line[60:66], "PDB temperature factor") if len(line) >= 66 else None
            ),
            "element": _pdb_element(atom_name, line[76:78] if len(line) >= 78 else ""),
            "formal_charge": _pdb_formal_charge(line[78:80] if len(line) >= 80 else ""),
            "source_record": line,
        })
    columns = {name: tuple(row[name] for row in rows) for name in rows[0]}
    non_atom_records = tuple(
        line for line in lines if line[:6].strip().upper() not in {"ATOM", "HETATM"}
    )
    source_connectivity = _pdb_connectivity(lines)
    connectivity = _induced_connectivity(source_connectivity, columns["serial"])
    return _point_cloud(
        path, "pdb", [(row["x"], row["y"], row["z"]) for row in rows],
        columns["serial"], columns["element"], columns, coordinate_units,
        model=selected_model, available_models=available_models,
        unit_cell=_pdb_cell(lines), connectivity_records=connectivity,
        source_connectivity_records=source_connectivity,
        connectivity_selection_policy=(
            "source_record" if connectivity == source_connectivity
            else "induced_by_selected_atom_ids"
        ),
        non_atom_records=non_atom_records,
    )


def read_pdbqt(path, *, coordinate_units="angstrom", model=None):
    """Read one PDBQT coordinate model, retaining atom types and partial charges."""
    coordinate_units = _units(coordinate_units)
    lines = _read_lines(path)
    atom_lines, selected_model, available_models = _selected_pdb_atoms(lines, model)
    rows = []
    for line in atom_lines:
        if len(line) < 54:
            raise ValueError("PDBQT atom record is truncated")
        serial = line[6:11].strip()
        if not serial:
            raise ValueError("PDBQT atom record is missing its serial ID")
        atom_name = line[12:16]
        atom_type = line[77:].strip() if len(line) > 77 else ""
        if not atom_type:
            fields = line.split()
            atom_type = fields[-1] if fields else ""
        partial_charge = line[66:76].strip() if len(line) >= 76 else ""
        if not partial_charge:
            fields = line.split()
            partial_charge = fields[-2] if len(fields) >= 2 else ""
        rows.append({
            "record_name": line[:6].strip().upper(),
            "serial": _id(serial),
            "atom_name": atom_name.strip(),
            "alternate_location": line[16:17].strip() or None,
            "residue_name": line[17:20].strip() or None,
            "chain_id": line[21:22].strip() or None,
            "residue_sequence": _id(line[22:26]) if line[22:26].strip() else None,
            "insertion_code": line[26:27].strip() or None,
            "x": _float(line[30:38], "PDBQT x coordinate"),
            "y": _float(line[38:46], "PDBQT y coordinate"),
            "z": _float(line[46:54], "PDBQT z coordinate"),
            "occupancy": _optional_float(line[54:60], "PDBQT occupancy") if len(line) >= 60 else None,
            "temperature_factor": (
                _optional_float(line[60:66], "PDBQT temperature factor") if len(line) >= 66 else None
            ),
            "partial_charge": _float(partial_charge, "PDBQT partial charge"),
            "atom_type": atom_type,
            "element": _pdbqt_element(atom_type, atom_name),
            "source_record": line,
        })
    columns = {name: tuple(row[name] for row in rows) for name in rows[0]}
    non_atom_records = tuple(
        line for line in lines if line[:6].strip().upper() not in {"ATOM", "HETATM"}
    )
    return _point_cloud(
        path, "pdbqt", [(row["x"], row["y"], row["z"]) for row in rows],
        columns["serial"], columns["element"], columns, coordinate_units,
        model=selected_model, available_models=available_models,
        charge_type="partial", non_atom_records=non_atom_records,
    )


def _mol2_sections(lines):
    sections = {}
    current = None
    for line in lines:
        if line.upper().startswith("@<TRIPOS>"):
            current = line[len("@<TRIPOS>"):].strip().upper()
            if current in sections:
                raise ValueError(f"MOL2 contains repeated {current!r} sections")
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def read_mol2(path, *, coordinate_units="angstrom"):
    """Read one Tripos MOL2 molecule and preserve its atom and bond records."""
    coordinate_units = _units(coordinate_units)
    sections = _mol2_sections(_read_lines(path))
    if "MOLECULE" not in sections or "ATOM" not in sections:
        raise ValueError("MOL2 requires MOLECULE and ATOM sections")
    molecule_values = [line.strip() for line in sections["MOLECULE"]
                       if line.strip() and not line.lstrip().startswith("#")]
    if len(molecule_values) < 2:
        raise ValueError("MOL2 MOLECULE header is incomplete")
    counts = molecule_values[1].split()
    if not counts:
        raise ValueError("MOL2 atom count is missing")
    try:
        declared_atom_count = int(counts[0])
        declared_bond_count = int(counts[1]) if len(counts) > 1 else 0
    except ValueError as exc:
        raise ValueError("MOL2 molecule counts must be integers") from exc
    rows = []
    for line in sections["ATOM"]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 6:
            raise ValueError("MOL2 atom record requires ID, name, x, y, z, and type")
        charge = _float(fields[8], "MOL2 partial charge") if len(fields) >= 9 else None
        rows.append({
            "atom_id": _id(fields[0]),
            "atom_name": fields[1],
            "x": _float(fields[2], "MOL2 x coordinate"),
            "y": _float(fields[3], "MOL2 y coordinate"),
            "z": _float(fields[4], "MOL2 z coordinate"),
            "atom_type": fields[5],
            "element": _canonical_element(fields[5].split(".", 1)[0]),
            "substructure_id": _id(fields[6]) if len(fields) >= 7 else None,
            "substructure_name": fields[7] if len(fields) >= 8 else None,
            "partial_charge": charge,
            "status_bits": tuple(fields[9:]),
            "source_record": line,
        })
    if declared_atom_count != len(rows) or declared_atom_count <= 0:
        raise ValueError("MOL2 declared atom count does not match its ATOM section")
    atom_ids = {row["atom_id"] for row in rows}
    bonds = []
    for line in sections.get("BOND", ()):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 4:
            raise ValueError("MOL2 bond record requires ID, two atom IDs, and type")
        origin, target = _id(fields[1]), _id(fields[2])
        if origin not in atom_ids or target not in atom_ids:
            raise ValueError("MOL2 bond refers to an unknown atom ID")
        bonds.append({
            "id": _id(fields[0]), "atoms": (origin, target),
            "type": fields[3], "attributes": tuple(fields[4:]),
        })
    if declared_bond_count != len(bonds):
        raise ValueError("MOL2 declared bond count does not match its BOND section")
    columns = {name: tuple(row[name] for row in rows) for name in rows[0]}
    molecule = {
        "name": molecule_values[0],
        "counts": tuple(_id(value) for value in counts),
        "molecule_type": molecule_values[2] if len(molecule_values) >= 3 else None,
        "charge_type": molecule_values[3] if len(molecule_values) >= 4 else None,
        "additional_lines": tuple(molecule_values[4:]),
    }
    other_sections = {
        name: tuple(values) for name, values in sections.items()
        if name not in {"MOLECULE", "ATOM", "BOND"}
    }
    return _point_cloud(
        path, "mol2", [(row["x"], row["y"], row["z"]) for row in rows],
        columns["atom_id"], columns["element"], columns, coordinate_units,
        molecule=molecule,
        declared_counts={"atoms": declared_atom_count, "bonds": declared_bond_count},
        bonds=tuple(bonds), other_sections=other_sections,
        charge_type=molecule["charge_type"],
    )


_MDL_CHARGE_CODES = {0: 0, 1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}


def _mdl_v2000(lines, counts_index):
    counts_line = lines[counts_index]
    try:
        atom_count = int(counts_line[0:3])
        bond_count = int(counts_line[3:6])
    except ValueError as exc:
        raise ValueError("V2000 atom and bond counts must be integers") from exc
    if atom_count <= 0 or bond_count < 0:
        raise ValueError("V2000 requires a positive atom count and a nonnegative bond count")
    if len(lines) < counts_index + 1 + atom_count + bond_count:
        raise ValueError("V2000 coordinate or bond block is incomplete")
    atom_lines = lines[counts_index + 1:counts_index + 1 + atom_count]
    rows = []
    for index, line in enumerate(atom_lines, start=1):
        if len(line) < 34:
            raise ValueError("V2000 atom record is truncated")
        try:
            charge_code = int(line[36:39].strip() or 0) if len(line) >= 39 else 0
        except ValueError as exc:
            raise ValueError("V2000 atom charge code must be an integer") from exc
        rows.append({
            "atom_id": index,
            "element": _canonical_element(line[31:34]),
            "x": _float(line[0:10], "V2000 x coordinate"),
            "y": _float(line[10:20], "V2000 y coordinate"),
            "z": _float(line[20:30], "V2000 z coordinate"),
            "mass_difference": (
                _integer(line[34:36], "V2000 mass difference") if line[34:36].strip() else 0
            ),
            "formal_charge": _MDL_CHARGE_CODES.get(charge_code),
            "charge_code": charge_code,
            "atom_attributes": tuple(line[39:].split()) if len(line) > 39 else (),
            "source_record": line,
        })
    atom_ids = {row["atom_id"] for row in rows}
    bond_start = counts_index + 1 + atom_count
    bonds = []
    for bond_id, line in enumerate(lines[bond_start:bond_start + bond_count], start=1):
        if len(line) < 12:
            raise ValueError("V2000 bond record is truncated")
        try:
            origin, target, bond_type = int(line[0:3]), int(line[3:6]), int(line[6:9])
        except ValueError as exc:
            raise ValueError("V2000 bond fields must be integers") from exc
        if origin not in atom_ids or target not in atom_ids:
            raise ValueError("V2000 bond refers to an unknown atom ID")
        bonds.append({
            "id": bond_id, "atoms": (origin, target), "type": bond_type,
            "attributes": tuple(_id(value) for value in line[9:].split()),
        })
    all_property_lines = lines[bond_start + bond_count:]
    end_offset = next((index for index, line in enumerate(all_property_lines)
                       if line.startswith("M  END")), len(all_property_lines) - 1)
    property_lines = tuple(
        line for line in all_property_lines[:end_offset + 1] if line.startswith("M  ")
    )
    charges = {row["atom_id"]: row["formal_charge"] for row in rows}
    for line in property_lines:
        if not line.startswith("M  CHG"):
            continue
        fields = line.split()
        try:
            count = int(fields[2])
            values = fields[3:]
            if len(values) != count * 2:
                raise ValueError
            for offset in range(0, len(values), 2):
                atom_id, charge = int(values[offset]), int(values[offset + 1])
                if atom_id not in charges:
                    raise ValueError
                charges[atom_id] = charge
        except (IndexError, ValueError) as exc:
            raise ValueError("invalid V2000 M  CHG record") from exc
    for row in rows:
        row["formal_charge"] = charges[row["atom_id"]]
    return rows, tuple(bonds), property_lines, atom_count, bond_count


def _v30_payloads(lines):
    payloads = []
    pending = ""
    for line in lines:
        if not line.startswith("M  V30 "):
            continue
        payload = line[7:].rstrip()
        if payload.endswith("-"):
            pending += payload[:-1].rstrip() + " "
        else:
            payloads.append(pending + payload)
            pending = ""
    if pending:
        raise ValueError("unterminated V3000 continuation record")
    return payloads


def _v30_properties(tokens):
    properties = {}
    for token in tokens:
        if "=" in token:
            name, value = token.split("=", 1)
            properties[name.upper()] = value
    return properties


def _mdl_v3000(lines):
    payloads = _v30_payloads(lines)
    payload_names = {payload.upper() for payload in payloads}
    required_sections = {"BEGIN CTAB", "END CTAB", "BEGIN ATOM", "END ATOM"}
    if not required_sections <= payload_names:
        raise ValueError("V3000 CTAB and ATOM sections must be explicitly closed")
    counts = next((payload for payload in payloads if payload.upper().startswith("COUNTS ")), None)
    if counts is None:
        raise ValueError("V3000 CTAB is missing COUNTS")
    count_fields = shlex.split(counts)
    try:
        atom_count, bond_count = int(count_fields[1]), int(count_fields[2])
    except (IndexError, ValueError) as exc:
        raise ValueError("V3000 atom and bond counts must be integers") from exc
    if bond_count and not {"BEGIN BOND", "END BOND"} <= payload_names:
        raise ValueError("V3000 BOND section must be explicitly closed")
    section = None
    rows = []
    bonds = []
    for payload in payloads:
        upper = payload.upper()
        if upper == "BEGIN ATOM":
            section = "atom"
            continue
        if upper == "END ATOM":
            section = None
            continue
        if upper == "BEGIN BOND":
            section = "bond"
            continue
        if upper == "END BOND":
            section = None
            continue
        fields = shlex.split(payload)
        if section == "atom":
            if len(fields) < 6:
                raise ValueError("V3000 atom record is incomplete")
            properties = _v30_properties(fields[6:])
            rows.append({
                "atom_id": _integer(fields[0], "V3000 atom ID"),
                "element": _canonical_element(fields[1]),
                "x": _float(fields[2], "V3000 x coordinate"),
                "y": _float(fields[3], "V3000 y coordinate"),
                "z": _float(fields[4], "V3000 z coordinate"),
                "atom_map": _integer(fields[5], "V3000 atom map"),
                "formal_charge": (
                    _integer(properties["CHG"], "V3000 formal charge")
                    if "CHG" in properties else 0
                ),
                "atom_properties": tuple(fields[6:]),
                "source_record": payload,
            })
        elif section == "bond":
            if len(fields) < 4:
                raise ValueError("V3000 bond record is incomplete")
            bonds.append({
                "id": _integer(fields[0], "V3000 bond ID"),
                "type": _integer(fields[1], "V3000 bond type"),
                "atoms": (
                    _integer(fields[2], "V3000 bond atom ID"),
                    _integer(fields[3], "V3000 bond atom ID"),
                ),
                "attributes": tuple(fields[4:]),
            })
    if atom_count <= 0 or len(rows) != atom_count or len(bonds) != bond_count:
        raise ValueError("V3000 declared counts do not match its atom and bond blocks")
    atom_ids = {row["atom_id"] for row in rows}
    if any(origin not in atom_ids or target not in atom_ids
           for bond in bonds for origin, target in (bond["atoms"],)):
        raise ValueError("V3000 bond refers to an unknown atom ID")
    mdl_end = next(index for index, line in enumerate(lines) if line.startswith("M  END"))
    property_lines = tuple(
        line for line in lines[:mdl_end + 1]
        if line.startswith("M  ") and not line.startswith("M  V30 ")
    )
    return rows, tuple(bonds), property_lines, atom_count, bond_count


def _sdf_properties(lines):
    collected = {}
    index = 0
    while index < len(lines):
        match = _SDF_FIELD.match(lines[index])
        if not match:
            index += 1
            continue
        name = match.group(1).strip()
        index += 1
        values = []
        while index < len(lines) and lines[index].strip() and lines[index] != "$$$$":
            values.append(lines[index])
            index += 1
        collected.setdefault(name, []).append("\n".join(values))
    return {
        name: entries[0] if len(entries) == 1 else tuple(entries)
        for name, entries in collected.items()
    }


def _read_mdl(path, format_name, coordinate_units):
    coordinate_units = _units(coordinate_units)
    lines = _read_lines(path)
    if format_name == "sdf":
        delimiters = [index for index, line in enumerate(lines) if line == "$$$$"]
        if len(delimiters) > 1 or (delimiters and any(line.strip() for line in lines[delimiters[0] + 1:])):
            raise ValueError("SDF reader accepts exactly one structure record")
        record_end = delimiters[0] if delimiters else len(lines)
        record_lines = lines[:record_end]
    else:
        if any(line == "$$$$" for line in lines):
            raise ValueError("MOL reader does not accept SDF record delimiters")
        record_lines = lines
    if len(record_lines) < 4:
        raise ValueError("MDL record requires three header lines and a counts line")
    counts_index = next(
        (index for index, line in enumerate(record_lines[:10]) if "V2000" in line or "V3000" in line),
        None,
    )
    if counts_index is None:
        raise ValueError("MDL counts line must declare V2000 or V3000")
    end_index = next((index for index, line in enumerate(record_lines)
                      if line.startswith("M  END")), None)
    if end_index is None:
        raise ValueError("MDL record is missing the required M  END terminator")
    version = "V3000" if "V3000" in record_lines[counts_index] else "V2000"
    if version == "V3000":
        rows, bonds, property_lines, atom_count, bond_count = _mdl_v3000(record_lines)
    else:
        rows, bonds, property_lines, atom_count, bond_count = _mdl_v2000(record_lines, counts_index)
    columns = {name: tuple(row[name] for row in rows) for name in rows[0]}
    trailing = record_lines[end_index + 1:]
    header = tuple((record_lines[:counts_index] + ["", "", ""])[:3])
    return _point_cloud(
        path, format_name, [(row["x"], row["y"], row["z"]) for row in rows],
        columns["atom_id"], columns["element"], columns, coordinate_units,
        ctfile_version=version,
        header={"name": header[0], "program": header[1], "comment": header[2]},
        declared_counts={"atoms": atom_count, "bonds": bond_count},
        bonds=bonds, mdl_property_records=property_lines,
        properties=_sdf_properties(trailing) if format_name == "sdf" else {},
    )


def read_mol(path, *, coordinate_units="angstrom"):
    """Read one MDL MOL V2000 or V3000 structure."""
    return _read_mdl(path, "mol", coordinate_units)


def read_sdf(path, *, coordinate_units="angstrom"):
    """Read one SDF structure and retain its data fields."""
    return _read_mdl(path, "sdf", coordinate_units)


def _cif_tokens(lines):
    tokens = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        if line.startswith(";"):
            content = [line[1:]] if len(line) > 1 else []
            line_index += 1
            while line_index < len(lines) and not lines[line_index].startswith(";"):
                content.append(lines[line_index])
                line_index += 1
            if line_index == len(lines):
                raise ValueError("unterminated CIF semicolon text field")
            tokens.append(_CifToken("\n".join(content), quoted=True))
            line_index += 1
            continue
        cursor = 0
        while cursor < len(line):
            while cursor < len(line) and line[cursor].isspace():
                cursor += 1
            if cursor == len(line) or line[cursor] == "#":
                break
            if line[cursor] in {"'", '"'}:
                quote = line[cursor]
                cursor += 1
                start = cursor
                while cursor < len(line):
                    closes_value = (
                        line[cursor] == quote
                        and (cursor + 1 == len(line) or line[cursor + 1].isspace())
                    )
                    if closes_value:
                        break
                    cursor += 1
                if cursor == len(line):
                    raise ValueError("unterminated quoted CIF value")
                tokens.append(_CifToken(line[start:cursor], quoted=True))
                cursor += 1
            else:
                start = cursor
                while cursor < len(line) and not line[cursor].isspace():
                    cursor += 1
                tokens.append(_CifToken(line[start:cursor]))
        line_index += 1
    return tokens


def _parse_cif(lines):
    tokens = _cif_tokens(lines)
    blocks = []
    block = None
    index = 0
    reserved = {"loop_", "stop_", "global_"}
    while index < len(tokens):
        token = tokens[index]
        lower = token.lower()
        if not token.quoted and lower.startswith("data_"):
            if block is not None:
                blocks.append(block)
            block = {"name": str(token[5:]), "tags": {}, "loops": [], "data_names": set()}
            index += 1
            continue
        if block is None:
            index += 1
            continue
        if not token.quoted and lower == "loop_":
            index += 1
            headers = []
            while (index < len(tokens) and not tokens[index].quoted
                   and tokens[index].startswith("_")):
                headers.append(str(tokens[index]))
                index += 1
            if not headers:
                raise ValueError("CIF loop has no data names")
            normalized_headers = tuple(_cif_key(header) for header in headers)
            if len(set(normalized_headers)) != len(normalized_headers):
                raise ValueError("CIF loop contains duplicate case-normalized data names")
            if set(normalized_headers) & block["data_names"]:
                raise ValueError("CIF data names must be unique within a data block")
            block["data_names"].update(normalized_headers)
            values = []
            while index < len(tokens):
                candidate = tokens[index]
                candidate_lower = candidate.lower()
                if not candidate.quoted and (
                    candidate.startswith("_") or candidate_lower in reserved
                    or candidate_lower.startswith("data_") or candidate_lower.startswith("save_")
                ):
                    break
                values.append(candidate)
                index += 1
            if len(values) % len(headers):
                raise ValueError("CIF loop values do not align with its data names")
            rows = tuple(tuple(values[offset:offset + len(headers)])
                         for offset in range(0, len(values), len(headers)))
            block["loops"].append({"headers": tuple(headers), "rows": rows})
            continue
        if not token.quoted and token.startswith("_"):
            if index + 1 >= len(tokens):
                raise ValueError(f"CIF data name {token!r} is missing its value")
            value = tokens[index + 1]
            value_lower = value.lower()
            if not value.quoted and (
                value.startswith("_") or value_lower in reserved
                or value_lower.startswith("data_") or value_lower.startswith("save_")
            ):
                raise ValueError(f"CIF data name {token!r} is missing its value")
            normalized_name = _cif_key(token)
            if normalized_name in block["data_names"]:
                raise ValueError("CIF data names must be unique within a data block")
            block["data_names"].add(normalized_name)
            block["tags"][token.lower()] = value
            index += 2
            continue
        index += 1
    if block is not None:
        blocks.append(block)
    if not blocks:
        raise ValueError("CIF contains no data_ block")
    atom_blocks = []
    for candidate in blocks:
        for loop in candidate["loops"]:
            names = {_cif_key(header) for header in loop["headers"]}
            cartesian = {"_atom_site_cartn_x", "_atom_site_cartn_y", "_atom_site_cartn_z"}
            fractional = {"_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"}
            if cartesian <= names or fractional <= names:
                atom_blocks.append((candidate, loop))
    if len(atom_blocks) != 1:
        raise ValueError("CIF must contain exactly one atom-site coordinate loop")
    return atom_blocks[0]


def _cif_float(value, context):
    if getattr(value, "quoted", False):
        raise ValueError(f"{context} is not an unquoted CIF number")
    if value in {".", "?"}:
        raise ValueError(f"{context} is missing")
    match = _CIF_NUMBER.fullmatch(value)
    if not match:
        raise ValueError(f"{context} is not a CIF number")
    return _float(match.group(1), context)


def _fractional_matrix(tags):
    names = {
        "length_a": "_cell_length_a", "length_b": "_cell_length_b",
        "length_c": "_cell_length_c", "angle_alpha": "_cell_angle_alpha",
        "angle_beta": "_cell_angle_beta", "angle_gamma": "_cell_angle_gamma",
    }
    missing = [tag for tag in names.values() if tag not in tags]
    if missing:
        raise ValueError(f"fractional CIF coordinates require cell parameters: {missing}")
    cell = {name: _cif_float(tags[tag], f"CIF {tag}") for name, tag in names.items()}
    a, b, c = cell["length_a"], cell["length_b"], cell["length_c"]
    angles = (cell["angle_alpha"], cell["angle_beta"], cell["angle_gamma"])
    if any(not 0 < angle < 180 for angle in angles):
        raise ValueError("CIF unit-cell angles must lie strictly between 0 and 180 degrees")
    alpha = math.radians(cell["angle_alpha"])
    beta = math.radians(cell["angle_beta"])
    gamma = math.radians(cell["angle_gamma"])
    sin_gamma = math.sin(gamma)
    if a <= 0 or b <= 0 or c <= 0 or abs(sin_gamma) <= 1e-15:
        raise ValueError("CIF unit cell is degenerate")
    cos_alpha, cos_beta, cos_gamma = math.cos(alpha), math.cos(beta), math.cos(gamma)
    gram_determinant = (
        1 + 2 * cos_alpha * cos_beta * cos_gamma
        - cos_alpha * cos_alpha - cos_beta * cos_beta - cos_gamma * cos_gamma
    )
    relative_tolerance = 1e-12
    if gram_determinant <= relative_tolerance:
        raise ValueError("CIF unit cell has zero or numerically degenerate volume")
    c_x = c * cos_beta
    c_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    c_z_squared = c * c - c_x * c_x - c_y * c_y
    if c_z_squared <= relative_tolerance * c * c:
        raise ValueError("CIF unit cell has zero or numerically degenerate volume")
    matrix = (
        (a, 0.0, 0.0),
        (b * cos_gamma, b * sin_gamma, 0.0),
        (c_x, c_y, math.sqrt(c_z_squared)),
    )
    return cell, matrix


def _fractional_to_cartesian(point, matrix):
    return tuple(sum(point[row] * matrix[row][column] for row in range(3))
                 for column in range(3))


def read_cif(path, *, coordinate_units="angstrom"):
    """Read one CIF atom-site loop, converting fractional sites through its cell.

    Symmetry-equivalent sites are intentionally not generated.  Raw atom-site
    values and the transformation matrix remain in metadata.
    """
    coordinate_units = _units(coordinate_units)
    block, atom_loop = _parse_cif(_read_lines(path))
    headers = atom_loop["headers"]
    name_lookup = {_cif_key(name): name for name in headers}
    columns = {
        name: tuple(row[index] for row in atom_loop["rows"])
        for index, name in enumerate(headers)
    }
    lower_columns = {_cif_key(name): values for name, values in columns.items()}
    normalized_tags = {_cif_key(name): value for name, value in block["tags"].items()}
    cell_names = {
        "_cell_length_a", "_cell_length_b", "_cell_length_c",
        "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
    }
    cell, matrix = (
        _fractional_matrix(normalized_tags) if cell_names <= normalized_tags.keys()
        else (None, None)
    )
    cartesian_names = ("_atom_site_cartn_x", "_atom_site_cartn_y", "_atom_site_cartn_z")
    fractional_names = ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")
    if all(name in lower_columns for name in cartesian_names):
        points = [tuple(_cif_float(lower_columns[name][row], f"CIF {name}")
                        for name in cartesian_names)
                  for row in range(len(atom_loop["rows"]))]
        coordinate_source = "cartesian"
    else:
        fractional_points = [tuple(_cif_float(lower_columns[name][row], f"CIF {name}")
                                   for name in fractional_names)
                             for row in range(len(atom_loop["rows"]))]
        if matrix is None:
            cell, matrix = _fractional_matrix(normalized_tags)
        points = [_fractional_to_cartesian(point, matrix) for point in fractional_points]
        coordinate_source = "fractional_converted_to_cartesian"
    id_name = next((name for name in ("_atom_site_label", "_atom_site_id")
                    if name in lower_columns), None)
    ids = tuple(
        str(value) if getattr(value, "quoted", False) else _id(value)
        for value in lower_columns[id_name]
    ) if id_name else tuple(
        range(1, len(points) + 1)
    )
    element_name = "_atom_site_type_symbol"
    if element_name in lower_columns:
        labels = tuple(_canonical_element(value) for value in lower_columns[element_name])
    elif id_name:
        # Site identifiers such as ``CA`` are not unambiguous element symbols
        # in macromolecular files.  Preserve them as labels without guessing.
        labels = tuple(lower_columns[id_name])
    else:
        labels = tuple("" for _ in points)
    # Keep exact CIF data names in the row-aligned columns mapping and also add
    # canonical element/charge accessors when the original names differ.
    columns["element"] = (
        labels if element_name in lower_columns else tuple(None for _ in points)
    )
    charge_name = next((name for name in (
        "_atom_site_charge", "_atom_site_formal_charge", "_atom_site_pdbx_formal_charge",
    )
                        if name in lower_columns), None)
    if charge_name:
        columns["formal_charge"] = tuple(
            None if not getattr(value, "quoted", False) and value in {".", "?"}
            else _cif_float(value, f"CIF {charge_name}")
            for value in lower_columns[charge_name]
        )
    other_loops = tuple(loop for loop in block["loops"] if loop is not atom_loop)
    return _point_cloud(
        path, "cif", points, ids, labels, columns, coordinate_units,
        data_block=block["name"], coordinate_source=coordinate_source,
        unit_cell=cell, fractional_to_cartesian_matrix=matrix,
        cif_tags=dict(block["tags"]),
        cif_normalized_tags=normalized_tags,
        cif_other_loops=other_loops,
        symmetry_expansion="not_applied", atom_site_id_column=name_lookup.get(id_name),
    )


__all__ = ["read_pdb", "read_mol2", "read_sdf", "read_mol", "read_pdbqt", "read_cif"]
