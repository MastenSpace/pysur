"""
    Copyright 2017 Masten Space Systems, Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Author: Jack Nelson <jnelson@masten.aero>
"""
import sys
import os
import math
import pickle

class Pipeline(object):
    """
        Class representing a surrogate modeling and design optimization pipeline. An instance of sm_pipe will 
        handle the discrete stages of the pipeline, known as filters, as well as the interfaces between
        the filters, known as pipes.


    """

    def __init__(self, name, rootdir, filters = [], data_sources = []):
        """
            Initialize a pipeline instance. Choose a root data directory, and add filter and pipe instances.
        """
        self.name = name
        self.rootdir = rootdir
        self.filters = filters
        self.data_sources = data_sources

    def add_filter(self, filter_, upstream, downstream):
        """
            Add a filter to the pipeline and specify the upstream and downstream filters.
        """
        self.filters.append(filter_)
        filter_.upstream = upstream
        filter_.downstream = downstream

        return self.filters

    def remove_filter(self, filter_):
        """
            Remove a filter from the pipeline.
        """
        self.filters.remove(filter_)
        if not any(item == filter_ for item in self.filters):
            return self.filters
        else:
            return None

    def get_filter_tags(self):
        return [filt.tag for filt in self.filters]

    def get_connections(self):
        cxn_list = []
        for filter_ in self.filters:
            cxn_list.append([filter_.upstream.tag, filter_.tag])
            cxn_list.append([filter_.tag, filter_.downstream.tag])

        return cxn_list

    def get_pipeline_overview(self):

        cxns = self.get_connections()


    def connect(self, upstream, downstream):
        """
            Connect two filters to each other logically.
        """
        upstream.downconnect(downstream)
        downstream.upconnect(upstream)

        return self.get_connections()


    def run_pipeline(self, filter_list = None):
        """
            Run the pipeline. If a list of filters is passed, only the filters in that list will be run. If
            no filter list is passed, all the filters in the pipeline will be run.

            Filters in the filter_list should be sequential, but they don't necessarily have to be, so long
            as the proper input files exist for each filter.
        """
        #if filter_list != None:
            #for filt in filter_list:
                #if filt in self.filters:



class Filter(object):
    """
        Abstract class for data filters in a pipeline. Should be subclassed elsewhere by classes
        which implement run_filter(). Filters work on data that is passed to them in a pipeline
        and can be linked in sequence or in parallel, i.e. a filter can receive input from
        multiple other sources, and can send output to multiple other sinks.

        This class is used to abstractly define the interface between multiple elements of a 
        pipeline. When creating a pipeline, each filter is separately instantiated, and then
        linked together using the upconnect() method for setting upstream connections (data
        sources), and using the downconnect() method for setting downstream connections (data
        sinks).
    """
    def __init__(self, tag, upstream = [], downstream = []):
        """
            Basic initialization of a filter instance. Must give each filter a unique tag.

            Initialize an upstream and downstream filter if provided. upstream and downstream
            are iterables of instances of other filters.
        """
        self.tag = tag
        self.upstream = upstream
        self.downstream = downstream    

    def downconnect(self, downstream):
        """
            Connect a filter downstream of this filter.
        """
        self.downstream.append(downstream)
        downstream()

    def upconnect(self, upstream):
        """
            Connect a filter upstream of this filter.
        """
        self.upstream.append(upstream)

    def run_filter(self, *args, **kwargs):
        return -1
