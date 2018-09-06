__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
import os
import numpy as np
import inspect


class TimestepInjections(object):
    def __init__(self, timestep_id, loads_p, loads_q, prods_p, prods_v, maintenance):
        self.id = timestep_id

        # Prods container
        self.prods_p = prods_p
        self.prods_v = prods_v
        # Loads container
        self.loads_p = loads_p
        self.loads_q = loads_q

        self.maintenance = maintenance

    def get_prods_p(self):
        return self.prods_p

    def get_prods_v(self):
        return self.prods_v

    def get_loads_q(self):
        return self.loads_q

    def get_loads_p(self):
        return self.loads_p

    def get_id(self):
        return self.id

    def get_maintenance(self):
        return self.maintenance


class Chronic(object):
    def __init__(self, source_folder):
        if not os.path.exists(source_folder):
            raise ValueError('Source folder %s does not exist' % source_folder)
        self.source_folder = source_folder

        # Containers for the ROI files paths
        self.fpath_loads_p = None
        self.fpath_loads_q = None
        self.fpath_prods_p = None
        self.fpath_prods_v = None
        self.fpath_ids = None
        self.fpath_imaps = None
        self.fpath_maintenance = None

        # Containers for the productions and loads data
        self.prods_p = None
        self.prods_v = None
        self.loads_p = None
        self.loads_q = None

        self.imaps = None
        self.maintenance = None

        self.timestep_ids = None

        # Overall ordered container for the Scenarios
        self.timesteps_injections = []

        # Retrieve the input files of the chronic
        self.retrieve_input_files()
        data = self.retrieve_data()
        # Save data into self containers
        self.import_data(data)

        # Create scenarios based on the imported values
        self.construct_timesteps_injections()

    def retrieve_input_files(self):
        # Retrieve all the csv files within the folder
        csv_files = [f for f in os.listdir(self.source_folder) if
                     os.path.isfile(os.path.join(self.source_folder, f)) and f.endswith('.csv')]

        # Expected loads and prods files name
        fname_loads_p = '_N_loads_p.csv'
        fname_loads_q = '_N_loads_q.csv'
        fname_prods_p = '_N_prod_p.csv'
        fname_prods_v = '_N_prod_v.csv'
        # Expected ID file name
        fname_ids = '_N_simu_ids.csv'
        fname_imaps = '_N_imaps.csv'
        # Maintenance and hazards
        fname_maintenance = 'maintenance.csv'

        mandatory_files = [fname_loads_p, fname_loads_q, fname_prods_p, fname_prods_v, fname_ids, fname_imaps,
                           fname_maintenance]
        # Check whether all mandatory files are present within the source directory
        for mandatory_file in mandatory_files:
            if mandatory_file not in csv_files:
                raise FileExistsError('File %s does not exist but is mandatory' % mandatory_file)

        # At this point, all the necesarry files are present within the source folder: save their absolute paths
        self.fpath_loads_p = os.path.join(self.source_folder, fname_loads_p)
        self.fpath_loads_q = os.path.join(self.source_folder, fname_loads_q)
        self.fpath_prods_p = os.path.join(self.source_folder, fname_prods_p)
        self.fpath_prods_v = os.path.join(self.source_folder, fname_prods_v)
        self.fpath_ids = os.path.join(self.source_folder, fname_ids)
        self.fpath_imaps = os.path.join(self.source_folder, fname_imaps)
        self.fpath_maintenance = os.path.join(self.source_folder, fname_maintenance)

    @staticmethod
    def get_csv_content(csv_absolute_fpath):
        return np.genfromtxt(csv_absolute_fpath, dtype=np.float32, delimiter=';', skip_header=True)

    def retrieve_data(self):
        # Keys: name of attribute; value: data of the associated attribute
        data = {}
        # Retrieve all the attributes of this class representing an input file (starts with 'fpath_')
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if a[0].startswith('fpath_')]

        for attribute_name, attribute_path in attributes:
            data[attribute_name] = self.get_csv_content(attribute_path)

        return data

    def import_data(self, data):
        # Save productions and loads data
        self.prods_p = data['fpath_prods_p']
        self.prods_v = data['fpath_prods_v']
        self.loads_p = data['fpath_loads_p']
        self.loads_q = data['fpath_loads_q']
        self.imaps = data['fpath_imaps'].tolist()

        self.maintenance = data['fpath_maintenance']

        # Scenarios ids
        self.timestep_ids = data['fpath_ids'].astype(np.int32).tolist()
        # Verify that all the ids are unique
        assert len(np.unique(self.timestep_ids)) == len(self.timestep_ids), 'There are timesteps with the same id'

    def construct_timesteps_injections(self):
        """
        Loop over all the pertinent data row by row creating scenarios that are stored within the self.scenarios
        container.
        """
        for scen_id, loads_p, loads_q, prods_p, prods_v, maintenance in zip(self.timestep_ids, self.loads_p,
                                                                            self.loads_q, self.prods_p, self.prods_v,
                                                                            self.maintenance):
            timestep_injections = TimestepInjections(scen_id, loads_p, loads_q, prods_p, prods_v, maintenance)
            self.timesteps_injections.append(timestep_injections)

    def get_timestep_injections(self, timestep_id):
        if timestep_id not in self.timestep_ids:
            raise ValueError('Could not find TimestepInjections with id', timestep_id)
        return self.timesteps_injections[self.timestep_ids.index(timestep_id)]

    def get_timestep_ids(self):
        return self.timestep_ids

    def get_imaps(self):
        return self.imaps
