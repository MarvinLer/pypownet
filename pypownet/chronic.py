__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
import os
import numpy as np
import inspect
import logging
from datetime import datetime


class TimestepEntries(object):
    def __init__(self, timestep_id, loads_p, loads_q, prods_p, prods_v, maintenance, hazards, date,
                 planned_loads_p=None, planned_loads_q=None, planned_prods_p=None, planned_prods_v=None):
        self.id = timestep_id

        # Prods and loads containers
        self.prods_p = prods_p
        self.prods_v = prods_v
        self.loads_p = loads_p
        self.loads_q = loads_q
        # Planned injections containers
        self.planned_prods_p = planned_prods_p
        self.planned_prods_v = planned_prods_v
        self.planned_loads_p = planned_loads_p
        self.planned_loads_q = planned_loads_q

        self.maintenance = maintenance
        self.hazards = hazards

        self.datetime = datetime.strptime(date.lower(), '%Y-%b-%d;%H:%M')

    def get_prods_p(self):
        return self.prods_p

    def get_prods_v(self):
        return self.prods_v

    def get_loads_q(self):
        return self.loads_q

    def get_loads_p(self):
        return self.loads_p

    def get_planned_prods_p(self):
        return self.planned_prods_p

    def get_planned_prods_v(self):
        return self.planned_prods_v

    def get_planned_loads_q(self):
        return self.planned_loads_q

    def get_planned_loads_p(self):
        return self.planned_loads_p

    def get_id(self):
        return self.id

    def get_maintenance(self):
        return self.maintenance

    def get_hazards(self):
        return self.hazards

    def get_datetime(self):
        return self.datetime


class Chronic(object):
    def __init__(self, source_folder, with_previsions=True):
        if not os.path.exists(source_folder):
            raise ValueError('Source folder %s does not exist' % source_folder)
        self.source_folder = source_folder
        self.name = os.path.basename(source_folder)
        self.with_previsions = with_previsions  # True will seek and load planned injections (in Obs and simulate)

        # Containers for the ROI files paths
        self.fpath_loads_p = None
        self.fpath_loads_q = None
        self.fpath_prods_p = None
        self.fpath_prods_v = None
        self.fpath_ids = None
        self.fpath_imaps = None
        self.fpath_maintenance = None
        self.fpath_hazards = None
        # ROI planned injections files
        self.fpath_loads_p_planned = None
        self.fpath_loads_q_planned = None
        self.fpath_prods_p_planned = None
        self.fpath_prods_v_planned = None

        self.datetimes_path = None

        # Containers for the productions and loads data
        self.prods_p = None
        self.prods_v = None
        self.loads_p = None
        self.loads_q = None
        self.prods_p_planned = None
        self.prods_v_planned = None
        self.loads_p_planned = None
        self.loads_q_planned = None

        self.imaps = None
        self.maintenance = None
        self.hazards = None

        self.timestep_ids = None
        self.datetimes = None

        # Overall ordered container for the Scenarios
        self.timesteps_entries = []

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

        # Realized loads and prods files name
        fname_loads_p = '_N_loads_p.csv'
        fname_loads_q = '_N_loads_q.csv'
        fname_prods_p = '_N_prods_p.csv'
        fname_prods_v = '_N_prods_v.csv'
        # Planned loads and prods files name
        fname_loads_p_planned = '_N_loads_p_planned.csv'
        fname_loads_q_planned = '_N_loads_q_planned.csv'
        fname_prods_p_planned = '_N_prods_p_planned.csv'
        fname_prods_v_planned = '_N_prods_v_planned.csv'
        # Expected ID file name
        fname_ids = '_N_simu_ids.csv'
        fname_datetimes = '_N_datetimes.csv'
        fname_imaps = '_N_imaps.csv'
        # Maintenance and hazards
        fname_maintenance = 'maintenance.csv'
        fname_hazards = 'hazards.csv'

        mandatory_files = [fname_loads_p, fname_loads_q, fname_prods_p, fname_prods_v, fname_ids, fname_imaps,
                           fname_maintenance, fname_hazards, fname_datetimes]

        if self.with_previsions:
            mandatory_files.extend([fname_loads_p_planned, fname_loads_q_planned,
                                    fname_prods_p_planned, fname_prods_v_planned])
        # Check whether all mandatory files are present within the source directory
        for mandatory_file in mandatory_files:
            if mandatory_file not in csv_files:
                raise FileExistsError('File %s does not exist but is mandatory' % mandatory_file)

        # At this point, all the necesarry files are present within the source folder: save their absolute paths
        self.fpath_loads_p = os.path.join(self.source_folder, fname_loads_p)
        self.fpath_loads_q = os.path.join(self.source_folder, fname_loads_q)
        self.fpath_prods_p = os.path.join(self.source_folder, fname_prods_p)
        self.fpath_prods_v = os.path.join(self.source_folder, fname_prods_v)
        self.fpath_loads_p_planned = os.path.join(self.source_folder, fname_loads_p_planned)
        self.fpath_loads_q_planned = os.path.join(self.source_folder, fname_loads_q_planned)
        self.fpath_prods_p_planned = os.path.join(self.source_folder, fname_prods_p_planned)
        self.fpath_prods_v_planned = os.path.join(self.source_folder, fname_prods_v_planned)
        self.fpath_ids = os.path.join(self.source_folder, fname_ids)
        self.fpath_imaps = os.path.join(self.source_folder, fname_imaps)
        self.fpath_maintenance = os.path.join(self.source_folder, fname_maintenance)
        self.fpath_hazards = os.path.join(self.source_folder, fname_hazards)
        self.datetimes_path = os.path.join(self.source_folder, fname_datetimes)

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
        self.prods_p_planned = data['fpath_prods_p_planned']
        self.prods_v_planned = data['fpath_prods_v_planned']
        self.loads_p_planned = data['fpath_loads_p_planned']
        self.loads_q_planned = data['fpath_loads_q_planned']
        self.imaps = data['fpath_imaps'].tolist()

        # Slip data from planned by 1
        self.prods_p_planned[:-1] = self.prods_p_planned[1:]
        self.prods_v_planned[:-1] = self.prods_v_planned[1:]
        self.loads_p_planned[:-1] = self.loads_p_planned[1:]
        self.loads_q_planned[:-1] = self.loads_q_planned[1:]
        # self.prods_p_planned = np.concatenate((self.prods_p_planned[1:], [0] * len(self.prods_p_planned[0])))
        # self.prods_v_planned = np.concatenate((self.prods_v_planned[1:], [0] * len(self.prods_v_planned[0])))
        # self.loads_p_planned = np.concatenate((self.loads_p_planned[1:], [0] * len(self.loads_p_planned[0])))
        # self.loads_q_planned = np.concatenate((self.loads_q_planned[1:], [0] * len(self.loads_q_planned[0])))

        self.maintenance = data['fpath_maintenance']
        self.hazards = data['fpath_hazards']

        # Scenarios ids
        self.timestep_ids = data['fpath_ids'].astype(np.int32).tolist()
        self.datetimes = open(self.datetimes_path, 'r').read().splitlines()[1:]
        # Verify that all the ids are unique
        assert len(np.unique(self.timestep_ids)) == len(self.timestep_ids), 'There are timesteps with the same id'

    def construct_timesteps_injections(self):
        """
        Loop over all the pertinent data row by row creating scenarios that are stored within the self.scenarios
        container.
        """
        for scen_id, loads_p, loads_q, prods_p, prods_v, \
            planned_loads_p, planned_loads_q, planned_prods_p, planned_prods_v, \
            maintenance, hazards, date in zip(self.timestep_ids, self.loads_p, self.loads_q, self.prods_p, self.prods_v,
                                              self.loads_p_planned, self.loads_q_planned, self.prods_p_planned,
                                              self.prods_v_planned, self.maintenance, self.hazards, self.datetimes):
            timestep_entries = TimestepEntries(scen_id, loads_p, loads_q, prods_p, prods_v, maintenance, hazards, date,
                                               planned_loads_p, planned_loads_q, planned_prods_p, planned_prods_v)
            self.timesteps_entries.append(timestep_entries)

    def get_timestep_entries(self, timestep_id):
        if timestep_id not in self.timestep_ids:
            raise ValueError('Could not find TimestepInjections with id', timestep_id)
        return self.timesteps_entries[self.timestep_ids.index(timestep_id)]

    def get_planned_maintenance(self, timestep_id, horizon):
        index_begin_timestep = self.timestep_ids.index(timestep_id)
        timesteps_entries = self.timesteps_entries[index_begin_timestep:index_begin_timestep + horizon]
        maintenances = np.asarray([entry.get_maintenance() for entry in timesteps_entries])

        # Construct a vector of the timesteps before a maintenance will start on each line; 0 for no planned maintenance
        planned_maintenance = (maintenances != 0).argmax(axis=0)
        return planned_maintenance

    def get_timestep_duration(self):
        first_datetime = self.timesteps_entries[0].get_datetime()
        second_datetime = self.timesteps_entries[1].get_datetime()
        return (second_datetime - first_datetime).total_seconds()

    def get_timestep_ids(self):
        return self.timestep_ids

    def get_imaps(self):
        return self.imaps


class ChronicLooper(object):
    def __init__(self, chronics_folder, game_level, start_id, looping_mode):
        self.logger = logging.getLogger('pypownet.' + __name__)

        self.chronics_folder = os.path.abspath(chronics_folder)
        if not os.path.exists(self.chronics_folder):
            raise FileNotFoundError('Chronic folder %s does not exist' % self.chronics_folder)

        if looping_mode not in ['natural', 'random', 'fixed']:
            raise ValueError('Either "natural" mode (loops in the order of chronics ids), "random" (loops randomly) or'
                             '"fixed" (plays the same chronic)')
        self.looping_mode = looping_mode

        # Seeks all available chronics (sorts alphabetically)
        self.chronics = sorted([os.path.join(self.chronics_folder, d) for d in os.listdir(self.chronics_folder)
                                if not os.path.isfile(os.path.join(self.chronics_folder, d))])
        self.logger.info('Found %d chronics of game level %s; looping with mode %s starting with chronic %s' % (
            len(self.chronics), os.path.basename(self.chronics_folder), self.looping_mode,
            os.path.basename(self.chronics[start_id])))

        self.next_chronic_id = start_id if self.looping_mode != 'random' else np.random.choice(len(self.chronics))
        self.current_chronic_name = None

    def get_next_chronic_folder(self):
        res_chronic = self.chronics[self.next_chronic_id]
        self.current_chronic_name = os.path.basename(res_chronic)
        if self.looping_mode == 'natural':
            self.next_chronic_id = (self.next_chronic_id + 1) % len(self.chronics)
        elif self.looping_mode == 'random':
            self.next_chronic_id = np.random.choice(len(self.chronics))
        elif self.looping_mode == 'fixed':
            self.next_chronic_id = self.next_chronic_id
        return res_chronic

    def get_current_chronic_name(self):
        return self.current_chronic_name
