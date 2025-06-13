import h5py
import numpy as np
import os
import re

SOURCE_TAGS = ['CE', 'CD']
SOURCE_TAG_TO_ONEHOT = {
    tag: np.eye(len(SOURCE_TAGS), dtype=np.int8)[i]
    for i, tag in enumerate(SOURCE_TAGS)
}

def extract_suffix(group_name):
    """ Extract the last part of the group name after the last ':' """
    return group_name.split(":")[-1]  # Extract suffix after the last ':'

def copy_target_values(src_group, dest_group):
    if "target_values" in src_group and "target_values" not in dest_group:
        src_target_group = src_group["target_values"]
        dest_target_group = dest_group.create_group("target_values")

        for dataset_name in src_target_group:
            dataset = src_target_group[dataset_name]
            data = dataset[()]  
            dest_target_group.create_dataset(dataset_name, data=data, dtype=dataset.dtype)

def merge_node_features(src_group, dest_group, existing_nodes):
    """
    Merge `node_features`, using `_name` as the unique Node ID.
    Skip rows in other datasets if `_name` is duplicated.
    """
    if "_name" not in src_group:
        print(f"Skipping node_features (No `_name` found)")
        return

    src_names = np.array([n.encode("utf-8") if isinstance(n, str) else n for n in src_group["_name"][:]])
    dest_names = np.array([n.encode("utf-8") if isinstance(n, str) else n for n in dest_group["_name"][:]]) if "_name" in dest_group else np.array([])

    # Identify new node indices that are not in existing_nodes
    new_indices = [i for i, name in enumerate(src_names) if name not in existing_nodes]

    if not new_indices:
        print(f"Skipping node_features (All `_name` values already exist)")
        return

    # Add new `_name` values to the existing set
    existing_nodes.update(src_names[new_indices])

    for dataset_name in src_group:
        if dataset_name == "_name":  # Merge `_name` dataset
            if dataset_name in dest_group:
                dest_dataset = dest_group[dataset_name]
                dest_dataset.resize((dest_dataset.shape[0] + len(new_indices)), axis=0)
                dest_dataset[-len(new_indices):] = src_names[new_indices]
            else:
                dt = h5py.special_dtype(vlen=bytes)  # Variable-length UTF-8
                dest_group.create_dataset(dataset_name, data=src_names[new_indices], dtype=dt, maxshape=(None,), chunks=True)
        else:  # Merge other datasets while skipping duplicate rows
            data = src_group[dataset_name][:][new_indices]

            if dataset_name in dest_group:
                dest_dataset = dest_group[dataset_name]
                dest_dataset.resize((dest_dataset.shape[0] + len(new_indices)), axis=0)
                dest_dataset[-len(new_indices):] = data
            else:
                maxshape = (None,) + data.shape[1:]
                dest_group.create_dataset(dataset_name, data=data, maxshape=maxshape, chunks=True)

def merge_edge_features(src_group, dest_group, source_tag, seen_edge_names):
    """
    Merge `edge_features` without checking `_name`.
    """
    edge_names_raw = src_group["_name"][:]
    edge_names = np.array([n.decode("utf-8") if isinstance(n, bytes) else n for n in edge_names_raw])

    new_mask = np.array([name not in seen_edge_names for name in edge_names])
    new_indices = np.where(new_mask)[0]
    
    seen_edge_names.update(edge_names[new_indices])
    
    edge_len = len(new_indices)
    onehot = SOURCE_TAG_TO_ONEHOT.get(source_tag, np.zeros(len(SOURCE_TAGS), dtype=np.int8))
    onehot_matrix = np.repeat(onehot[np.newaxis, :], edge_len, axis=0)  # shape: (E, 3)
    
    for dataset_name in src_group:
        full_data = src_group[dataset_name][()]
        data = full_data[new_indices]

        if dataset_name in dest_group:
            dest_dataset = dest_group[dataset_name]
            dest_dataset.resize((dest_dataset.shape[0] + data.shape[0]), axis=0)
            dest_dataset[-data.shape[0]:] = data
        else:
            maxshape = (None,) + data.shape[1:] if len(data.shape) > 1 else (None,)
            # ? Ensure `_name` is stored as variable-length UTF-8
            dt = h5py.special_dtype(vlen=str) if dataset_name == "_name" else data.dtype
            dest_group.create_dataset(dataset_name, data=data, dtype=dt, maxshape=maxshape, chunks=True)
            
    if "source_file" in dest_group:
        dset = dest_group["source_file"]
        dset.resize((dset.shape[0] + onehot_matrix.shape[0], onehot_matrix.shape[1]))
        dset[-onehot_matrix.shape[0]:] = onehot_matrix
    else:
        dest_group.create_dataset("source_file", data=onehot_matrix, maxshape=(None, onehot_matrix.shape[1]), chunks=True)
        

def update_edge_index(h5_out):

    for group_name in list(h5_out.keys()):  # Convert keys to list to avoid runtime changes
        if not isinstance(group_name, str):
            group_name = group_name.decode("utf-8")  # Decode bytes to string if needed

        if "node_features" in h5_out[group_name] and "edge_features" in h5_out[group_name]:
            node_group = h5_out[group_name]["node_features"]
            edge_group = h5_out[group_name]["edge_features"]

            if "_name" not in edge_group or "_index" not in edge_group or "_name" not in node_group:
                print(f"??  Skipping edge index update in `{group_name}` (Missing datasets)")
                continue

            # Decode `_name` dataset
            edge_names = np.array([e.decode("utf-8") if isinstance(e, bytes) else e for e in edge_group["_name"][:]])
            node_names = np.array([n.decode("utf-8") if isinstance(n, bytes) else n for n in node_group["_name"][:]])

            # Create a mapping from node names to their row indices
            node_name_to_index = {name: i for i, name in enumerate(node_names)}

            updated_index = []
            invalid_names = []
            missing_nodes = []
            for edge_name in edge_names:
                if "-" in edge_name:
                    #node1, node2 = edge_name.split("-")
                    #if node1 in node_name_to_index and node2 in node_name_to_index:
                    #    updated_index.append([node_name_to_index[node1], node_name_to_index[node2]])
                    parts = re.split(r'-(?!\d)', edge_name)

                    # ? Ensure exactly two node names exist
                    if len(parts) == 2:
                        node1, node2 = parts
                        if node1 in node_name_to_index and node2 in node_name_to_index:
                            updated_index.append([node_name_to_index[node1], node_name_to_index[node2]])
                        else:
                            updated_index.append([-1, -1])
                            missing_nodes.append((node1, node2))
                    else:
                        invalid_names.append(edge_name)
                        print(edge_name)

            if invalid_names:
                print(f"??  Found {len(invalid_names)} invalid `_name` values in `{group_name}`:")
                for invalid in invalid_names:
                    print(f"   ? {invalid}")

            if missing_nodes:
                print(f"\n??  Missing nodes in `{group_name}/edge_features`:")
                for node1, node2 in missing_nodes:
                    print(f"   ? {node1}, {node2} not found in `node_features`")

            updated_index = np.array(updated_index, dtype=np.int64)

            # ? Remove rows where `_index` is `[-1, -1]` and delete those rows in all `edge_features` datasets
            valid_rows = ~np.all(updated_index == -1, axis=1)  # Keep only rows where `_index` is NOT `[-1, -1]`
            filtered_index = updated_index[valid_rows]

            if "_index" in edge_group:
                edge_group["_index"].resize((filtered_index.shape[0], 2))
                edge_group["_index"][:] = filtered_index

            # ? Remove matching rows in all `edge_features` datasets
            for dataset_name in edge_group:
                if dataset_name not in ["_index"]:  # ? Skip `_index` itself
                    data = edge_group[dataset_name][:]
                    filtered_data = data[valid_rows]

                    if filtered_data.shape[0] != data.shape[0]:  # ? Only update if rows were removed
                        edge_group[dataset_name].resize((filtered_data.shape[0],) + data.shape[1:])
                        edge_group[dataset_name][:] = filtered_data

            print(f"? Removed {np.sum(~valid_rows)} invalid rows from `{group_name}/edge_features`.")


def merge_hdf5_groups(src_group, dest_group, existing_nodes, source_tag, seen_edge_names):
    """
    Handle merging of `node_features` and `edge_features` separately.
    """
    for key in src_group:
        if isinstance(src_group[key], h5py.Group):
            if key == "node_features":  # Merge node_features, skipping duplicate `_name`
                if key not in dest_group:
                    dest_group.create_group(key)
                merge_node_features(src_group[key], dest_group[key], existing_nodes)
            elif key == "edge_features":  # Merge edge_features normally
                if key not in dest_group:
                    dest_group.create_group(key)
                merge_edge_features(src_group[key], dest_group[key], source_tag, seen_edge_names)
            elif key == "target_values":
                copy_target_values(src_group, dest_group)
            else:  # Recursively merge other groups
                if key not in dest_group:
                    dest_group.create_group(key)
                merge_hdf5_groups(src_group[key], dest_group[key], existing_nodes, source_tag)

def merge_hdf5_files(input_files, output_file):
    """
    Merge multiple HDF5 files with multi-level groups.
    `node_features`: Skip duplicate `_name` entries.
    `edge_features`: Append data normally.
    Use suffix in the top-level group name to determine merging.
    """
    group_map = {}  # Stores which suffix should be merged
    existing_nodes = set()  # Set to track merged Node IDs
    seen_edge_names = set()

    #  Step 1: Identify groups that should be merged
    for file in input_files:
        with h5py.File(file, 'r') as h5_in:
            for group_name in h5_in.keys():
                suffix = extract_suffix(group_name)  # Extract suffix
                if suffix not in group_map:
                    group_map[suffix] = group_name  # Set primary group name

    # Step 2: Merge data
    with h5py.File(output_file, 'w') as h5_out:
        for file in input_files:
            source_match = re.search(r'_(AC|CD|CE|AD|AE)', file)
            source_tag = source_match.group(1) if source_match else "UNK"
            
            with h5py.File(file, 'r') as h5_in:
                for group_name in h5_in.keys():
                    suffix = extract_suffix(group_name)
                    merged_group_name = group_map[suffix]  # Use primary group name

                    if merged_group_name not in h5_out:
                        h5_out.create_group(merged_group_name)  # Create group if not exists

                    merge_hdf5_groups(h5_in[group_name], h5_out[merged_group_name], existing_nodes, source_tag, seen_edge_names)

    with h5py.File(output_file, 'r+') as h5_out:
        update_edge_index(h5_out)


def check_hdf5_consistency(hdf5_file):
    """
    ? Check if all datasets in `node_features` have the same number of rows.
    ? Check if all datasets in `edge_features` have the same number of rows.
    ? Print warnings if inconsistencies are found.
    """
    with h5py.File(hdf5_file, 'r') as h5_out:
        for group_name in h5_out.keys():
            if "node_features" in h5_out[group_name]:
                node_group = h5_out[group_name]["node_features"]
                node_row_counts = {ds: node_group[ds].shape[0] for ds in node_group if len(node_group[ds].shape) > 0}

                if len(set(node_row_counts.values())) > 1:
                    print(f"??  Inconsistent row counts in `node_features` of `{group_name}`:")
                    for ds, rows in node_row_counts.items():
                        print(f"   ? {ds}: {rows} rows")

            if "edge_features" in h5_out[group_name]:
                edge_group = h5_out[group_name]["edge_features"]
                edge_row_counts = {ds: edge_group[ds].shape[0] for ds in edge_group if len(edge_group[ds].shape) > 0}

                if len(set(edge_row_counts.values())) > 1:
                    print(f"??  Inconsistent row counts in `edge_features` of `{group_name}`:")
                    for ds, rows in edge_row_counts.items():
                        print(f"   ? {ds}: {rows} rows")


def filter_edges_by_source(input_file, output_file, source_index_to_keep):
    """
    Parameters:
        input_file: str - path to the original merged HDF5
        output_file: str - new HDF5 file to write filtered graphs
        source_index_to_keep: int - index in one-hot [AC=0, CD=1, CE=2]
    """
    with h5py.File(input_file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
        for group_name in f_in.keys():
            group = f_in[group_name]

            edge_group = group['edge_features']
            edge_index = edge_group['_index'][()]  # shape (E, 2)
            source_file = edge_group['source_file'][()]  # shape (E, 3)

            # ? Filter edges by source_index
            mask = source_file[:, source_index_to_keep] == 1
            kept_edge_index = edge_index[mask]

            if len(kept_edge_index) == 0:
                continue

            # ? Determine used node IDs
            used_nodes = np.unique(kept_edge_index.flatten())
            node_id_map = {old: new for new, old in enumerate(used_nodes)}
            new_num_nodes = len(used_nodes)

            # ? Remap edge indices
            new_edge_index = np.array([[node_id_map[src], node_id_map[dst]] for src, dst in kept_edge_index])

            # ? Create new group
            out_group = f_out.create_group(group_name)
            node_out = out_group.create_group("node_features")
            edge_out = out_group.create_group("edge_features")

            # ? Copy & filter node features
            for name, dset in group['node_features'].items():
                data = dset[()]
                node_out.create_dataset(name, data=data[used_nodes])

            # ? Copy & filter edge features
            for name, dset in edge_group.items():
                if name == '_index':
                    edge_out.create_dataset('_index', data=new_edge_index, dtype=np.int64)
                elif name == 'source_file':
                    edge_out.create_dataset(name, data=source_file[mask])
                else:
                    edge_out.create_dataset(name, data=dset[()][mask])

            # ? Copy target_values if present
            if 'target_values' in group:
                target_out = out_group.create_group("target_values")
                for k, d in group['target_values'].items():
                    data = d[()] if d.shape == () else d[:]
                    target_out.create_dataset(k, data=data)

            print(f"? Filtered `{group_name}` with {len(new_edge_index)} edges and {new_num_nodes} nodes")


if __name__ == "__main__":
    input_files = ["data_processed_add_CE_8A/ppi/residue/proc-108920.hdf5", "data_processed_add_CD_8A/ppi/residue/proc-108921.hdf5"]
    output_file = "merged_output_CDE.hdf5"

    merge_hdf5_files(input_files, output_file)
    check_hdf5_consistency(output_file)

#filter_edges_by_source(
#    input_file="merged_output_all.hdf5",
#    output_file="merged_output_all.hdf5",
#    source_index_to_keep=0
#)
