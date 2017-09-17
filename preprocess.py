#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
'''
Workflow for doing preprocessing
that FMRIPREP doesn't complete, and derives standardized residuals from bold.
'''
# Things I should learn more about:
# regex
# Incorporate smarter way of getting bids data
# https://github.com/poldracklab/fmriprep/blob/master/fmriprep/utils/bids.py
from __future__ import print_function, division, absolute_import, unicode_literals
import argparse
import os
import re
from glob import glob
import gzip
from shutil import copy, copyfileobj
import uuid
from copy import deepcopy
from time import strftime
from bids.grabbids import BIDSLayout
import niworkflows.nipype.pipeline.engine as pe
from niworkflows.nipype import config as ncfg
from niworkflows.nipype.interfaces.utility import IdentityInterface
from niworkflows.nipype.interfaces.fsl import ImageStats, MultiImageMaths, SUSAN
from niworkflows.nipype.interfaces.fsl.utils import FilterRegressor
from niworkflows.nipype.interfaces.fsl.maths import MeanImage
from niworkflows.nipype.interfaces.utility import Function
# interface
from niworkflows.nipype import logging
from niworkflows.nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterfaceInputSpec,
    File, Directory, InputMultiPath, OutputMultiPath, Str
)

from niworkflows.interfaces.base import SimpleInterface

"""
################################################################
#########################   GLOBAL   ###########################
################################################################
"""
# taken from https://github.com/poldracklab/fmriprep/blob/master/fmriprep/interfaces/bids.py#L44
BIDS_NAME = re.compile(
    '^(.*\/)?(?P<subject_id>sub-[a-zA-Z0-9]+)(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
    '(_(?P<task_id>task-[a-zA-Z0-9]+))?(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
    '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?(_(?P<run_id>run-[a-zA-Z0-9]+))?')

# modified from: https://github.com/INCF/pybids/blob/master/bids/grabbids/config/bids.json
bids_deriv_config = {
                "entities": [
                    {
                        "name": "subject",
                        "pattern": "sub-([a-zA-Z0-9]+)",
                        "directory": "{{root}}/{subject}"
                    },
                    {
                        "name": "session",
                        "pattern": "ses-([a-zA-Z0-9]+)",
                        "mandatory": 'false',
                        "directory": "{{root}}/{subject}/{session}",
                        "missing_value": "ses-1"
                    },
                    {
                        "name": "run",
                        "pattern": "run-(\\d+)"
                    },
                    {
                        "name": "type",
                        "pattern": "[._]*([a-zA-Z0-9]*?)\\."
                    },
                    {
                        "name": "task",
                        "pattern": "task-(.*?)(?:_+)"
                    },
                    {
                        "name": "scans",
                        "pattern": "(.*\\_scans.tsv)$"
                    },
                    {
                        "name": "acquisition",
                        "pattern": "acq-(.*?)(?:_+)"
                    },
                    {
                        "name": "bval",
                        "pattern": "(.*\\.bval)$"
                    },
                    {
                        "name": "bvec",
                        "pattern": "(.*\\.bvec)$"
                    },
                    {
                        "name": "fmap",
                        "pattern": "(phasediff|magnitude[1-2]|phase[1-2]|fieldmap|epi)\\.nii"
                    },
                    {
                        "name": "modality",
                        "pattern": "[/\\\\](func|anat|fmap|dwi)[/\\\\]"
                    },
                    {
                        "name": "dir",
                        "pattern": "dir-([a-zA-Z0-9]+)"
                    },
                    {
                        "name": "acq",
                        "pattern": "acq-([a-zA-Z0-9]+)"
                    },
                    {
                        "name": "space",
                        "pattern": "space-([a-zA-Z0-9]+)"
                    },
                    {
                        "name": "variant",
                        "pattern": "variant-([a-zA-Z0-9]+)"
                    }
                ]
            }
"""
################################################################
################################################################
################################################################
"""

"""
################################################################
######################   Interfaces   ##########################
################################################################
"""


class FileNotFoundError(IOError):
    pass


class BIDSInfoInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, desc='input file, part of a BIDS tree')


class BIDSInfoOutputSpec(TraitedSpec):
    subject_id = traits.Str()
    session_id = traits.Str()
    task_id = traits.Str()
    acq_id = traits.Str()
    rec_id = traits.Str()
    run_id = traits.Str()


class BIDSInfo(SimpleInterface):
    """
    Extract metadata from a BIDS-conforming filename
    This interface uses only the basename, not the path, to determine the
    subject, session, task, run, acquisition or reconstruction.
    >>> from fmriprep.interfaces import BIDSInfo
    >>> from fmriprep.utils.bids import collect_data
    >>> bids_info = BIDSInfo()
    >>> bids_info.inputs.in_file = collect_data('ds114', '01')[0]['bold'][0]
    >>> bids_info.inputs.in_file  # doctest: +ELLIPSIS
    '.../ds114/sub-01/ses-retest/func/sub-01_ses-retest_task-covertverbgeneration_bold.nii.gz'
    >>> res = bids_info.run()
    >>> res.outputs
    <BLANKLINE>
    acq_id = <undefined>
    rec_id = <undefined>
    run_id = <undefined>
    session_id = ses-retest
    subject_id = sub-01
    task_id = task-covertverbgeneration
    <BLANKLINE>
    """
    input_spec = BIDSInfoInputSpec
    output_spec = BIDSInfoOutputSpec

    def _run_interface(self, runtime):
        match = BIDS_NAME.search(self.inputs.in_file)
        params = match.groupdict() if match is not None else {}
        self._results = {key: val for key, val in list(params.items())
                         if val is not None}
        return runtime


class BIDSDataGrabberInputSpec(BaseInterfaceInputSpec):
    subject_data = traits.Dict(Str, traits.Any)
    subject_id = Str()


class BIDSDataGrabberOutputSpec(TraitedSpec):
    out_dict = traits.Dict(desc='output data structure')
    preproc = OutputMultiPath(desc='output preproc functional images')
    mask = OutputMultiPath(desc='output masks for functional images')
    MELODICmix = OutputMultiPath(desc='output MELODICmix')
    AROMAnoiseICs = OutputMultiPath(desc='output AROMAnoiseICs')
    confounds = OutputMultiPath(desc='output confounds')


class BIDSDataGrabber(SimpleInterface):
    """
    Collect files from a BIDS directory structure
    >>> from fmriprep.interfaces import BIDSDataGrabber
    >>> from fmriprep.utils.bids import collect_data
    >>> bids_src = BIDSDataGrabber(anat_only=False)
    >>> bids_src.inputs.subject_data = collect_data('ds114', '01')[0]
    >>> bids_src.inputs.subject_id = 'ds114'
    >>> res = bids_src.run()
    >>> res.outputs.t1w  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['.../ds114/sub-01/ses-retest/anat/sub-01_ses-retest_T1w.nii.gz',
     '.../ds114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz']
    """
    input_spec = BIDSDataGrabberInputSpec
    output_spec = BIDSDataGrabberOutputSpec

    def __init__(self, *args, **kwargs):
        super(BIDSDataGrabber, self).__init__(*args, **kwargs)

    def _run_interface(self, runtime):
        bids_dict = self.inputs.subject_data

        self._results['out_dict'] = bids_dict
        self._results.update(bids_dict)

        if not bids_dict['preproc']:
            raise FileNotFoundError('No preproc images found for subject sub-{}'.format(
                self.inputs.subject_id))

        if not bids_dict['brainmask']:
            raise FileNotFoundError('No brainmasks found for subject sub-{}'.format(
                self.inputs.subject_id))

        if not bids_dict['MELODICmix']:
            raise FileNotFoundError('MELODICmix not found for subject sub-{}'.format(
                self.inputs.subject_id))

        if not bids_dict['AROMAnoiseICs']:
            raise FileNotFoundError('MELODICmix not found for subject sub-{}'.format(
                self.inputs.subject_id))

        if not bids_dict['confounds']:
            raise FileNotFoundError('confounds not found for subject sub-{}'.format(
                self.inputs.subject_id))

        for imtype in ['preproc', 'brainmask', 'MELODICmix', 'AROMAnoiseICs', 'confounds']:
            if not bids_dict[imtype]:
                raise ValueError('Could not not find some files for sub-{}'.format(self.inputs.subject_id))
                # LOGGER.warn('No \'{}\' images found for sub-{}'.format(
                # imtype, self.inputs.subject_id))

        return runtime


class DerivativesDataSinkInputSpec(BaseInterfaceInputSpec):
    base_directory = traits.Directory(
        desc='Path to the base directory for storing data.')
    in_file = InputMultiPath(File(exists=True), mandatory=True,
                             desc='the object to be saved')
    source_file = File(exists=False, mandatory=True, desc='the input func file')
    suffix = traits.Str('', mandatory=True, desc='suffix appended to source_file')
    extra_values = traits.List(traits.Str)


class DerivativesDataSinkOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File(exists=True, desc='written file path'))


class DerivativesDataSink(SimpleInterface):
    """
    Saves the `in_file` into a BIDS-Derivatives folder provided
    by `base_directory`, given the input reference `source_file`.
    >>> import tempfile
    >>> from fmriprep.utils.bids import collect_data
    >>> tmpdir = tempfile.mkdtemp()
    >>> tmpfile = os.path.join(tmpdir, 'a_temp_file.nii.gz')
    >>> open(tmpfile, 'w').close()  # "touch" the file
    >>> dsink = DerivativesDataSink(base_directory=tmpdir)
    >>> dsink.inputs.in_file = tmpfile
    >>> dsink.inputs.source_file = collect_data('ds114', '01')[0]['t1w'][0]
    >>> dsink.inputs.suffix = 'target-mni'
    >>> res = dsink.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../fmriprep/sub-01/ses-retest/anat/sub-01_ses-retest_T1w_target-mni.nii.gz'
    """
    input_spec = DerivativesDataSinkInputSpec
    output_spec = DerivativesDataSinkOutputSpec
    out_path_base = "NuisanceRegression"
    _always_run = True

    def __init__(self, out_path_base=None, **inputs):
        super(DerivativesDataSink, self).__init__(**inputs)
        self._results['out_file'] = []
        if out_path_base:
            self.out_path_base = out_path_base

    def _run_interface(self, runtime):
        src_fname, _ = _splitext(self.inputs.source_file)
        _, ext = _splitext(self.inputs.in_file[0])
        compress = ext == '.nii'
        if compress:
            ext = '.nii.gz'

        m = BIDS_NAME.search(src_fname)

        # TODO this quick and dirty modality detection needs to be implemented
        # correctly
        mod = 'func'

        base_directory = os.getcwd()
        if isdefined(self.inputs.base_directory):
            base_directory = os.path.abspath(self.inputs.base_directory)

        out_path = '{}/{subject_id}'.format(self.out_path_base, **m.groupdict())
        if m.groupdict().get('session_id') is not None:
            out_path += '/{session_id}'.format(**m.groupdict())
        out_path += '/{}'.format(mod)

        out_path = os.path.join(base_directory, out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        base_fname = os.path.join(out_path, src_fname)

        formatstr = '{bname}_{suffix}{ext}'
        if len(self.inputs.in_file) > 1 and not isdefined(self.inputs.extra_values):
            formatstr = '{bname}_{suffix}{i:04d}{ext}'

        for i, fname in enumerate(self.inputs.in_file):
            out_file = formatstr.format(
                bname=base_fname,
                suffix=self.inputs.suffix,
                i=i,
                ext=ext)
            if isdefined(self.inputs.extra_values):
                out_file = out_file.format(extra_value=self.inputs.extra_values[i])
            self._results['out_file'].append(out_file)
            if compress:
                with open(fname, 'rb') as f_in:
                    with gzip.open(out_file, 'wb') as f_out:
                        copyfileobj(f_in, f_out)
            else:
                copy(fname, out_file)

        return runtime


def _splitext(fname):
    fname, ext = os.path.splitext(os.path.basename(fname))
    if ext == '.gz':
        fname, ext2 = os.path.splitext(fname)
        ext = ext2 + ext
    return fname, ext


"""
################################################################
################################################################
################################################################
"""
"""
################################################################
#########################   UTILS   ############################
################################################################
"""


def collect_data(layout, participant_label, task=None, run=None, space=None):
    """
    Uses grabbids to retrieve the input data for a given participant
    >>> bids_root, _ = collect_data('ds054', '100185')
    >>> bids_root['fmap']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/fmap/sub-100185_magnitude1.nii.gz', \
'.../ds054/sub-100185/fmap/sub-100185_magnitude2.nii.gz', \
'.../ds054/sub-100185/fmap/sub-100185_phasediff.nii.gz']
    >>> bids_root['bold']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/func/sub-100185_task-machinegame_run-01_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-02_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-03_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-04_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-05_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-06_bold.nii.gz']
    >>> bids_root['sbref']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/func/sub-100185_task-machinegame_run-01_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-02_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-03_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-04_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-05_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-06_sbref.nii.gz']
    >>> bids_root['t1w']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/anat/sub-100185_T1w.nii.gz']
    >>> bids_root['t2w']  # doctest: +ELLIPSIS
    []
    """
    queries = {
        'preproc': {'subject': participant_label, 'modality': 'func', 'type': 'preproc',
                 'extensions': ['nii', 'nii.gz']},
        'brainmask': {'subject': participant_label, 'type': 'brainmask',
                  'extensions': ['nii', 'nii.gz']},
        'AROMAnoiseICs': {'subject': participant_label, 'modality': 'func', 'type': 'AROMAnoiseICs',
                'extensions': '.csv'},
        'MELODICmix': {'subject': participant_label, 'modality': 'func', 'type': 'MELODICmix',
                'extensions': 'tsv'},
        'confounds': {'subject': participant_label, 'modality': 'func', 'type': 'confounds',
                'extensions': 'tsv'},
    }

    if task:
        queries['preproc']['task'] = task
        queries['brainmask']['task'] = task
        queries['AROMAnoiseICs']['task'] = task
        queries['MELODICmix']['task'] = task
        queries['confounds']['task'] = task
    if run:
        queries['preproc']['run'] = run
        queries['brainmask']['run'] = run
        queries['AROMAnoiseICs']['run'] = run
        queries['MELODICmix']['run'] = run
        queries['confounds']['run'] = run
    if space:
        queries['preproc']['space'] = space
        queries['brainmask']['space'] = space


    return {modality: [x.filename for x in layout.get(**query)]
            for modality, query in queries.items()}

"""
################################################################
################################################################
################################################################
"""

"""
################################################################
##########################   BASE   ############################
################################################################
"""


def init_nuisance_regression_wf(confound_names, deriv_pipe_dir, low_pass,
                                subject_list, work_dir, result_dir,
                                ses_id, task_id, space, variant, res,
                                smooth, run_id,  regfilt, run_uuid, exclude_variant):
    r"""
    This workflow organizes the execution of preprocess, with a sub-workflow for
    each subject.

    Parameters

        confound_names : list of str or None
            Column names from FMRIPREP's confounds tsv that were selected for
            nuisance regression
        deriv_pipe_dir : str
            The absolute path to the FMRIPREP directory
        low_pass : float or None
            The frequency to set low pass filter (in Hz)
        subject_list : list of str
            List of subject labels
        work_dir : str
            The absolute path to execute workflows/nodes
        result_dir : str
            The absolute path to the base directory where results will be stored
        ses_id : str or None
            Session id to analyze
        task_id : str or None
            Task id to analyze
        space : str or None
            Output Space from FMRIPREP to analyze
        variant : str
            Output variant from FMRIPREP to analyze
        res : str
            Output resolution from FMRIPREP to analyze
        smooth : float or None
            smoothing kernel to apply to image
        run_id : str
            run number to analyze
        regfilt : bool
            Selects to run FilterRegressor from the output from ICA-AROMA
        run_uuid :
            Unique identifier for execution instance
    """

    nuisance_regression_wf = pe.Workflow(name='nuisance_regression_wf')
    # set where we put intermediate files/where we do processing
    nuisance_regression_wf.base_dir = os.path.join(work_dir, 'nuisance_regression_work')
    # get a representation of the directory/data structure
    layout = BIDSLayout(deriv_pipe_dir, config=bids_deriv_config)

    for subject_id in subject_list:
        subject_data = collect_data(layout,
                                    subject_id,
                                    task=task_id,
                                    run=run_id,
                                    space=space)
        # if you want to avoid using the ICA-AROMA variant
        if exclude_variant:
            subject_data['preproc'] = [preproc for preproc in subject_data['preproc'] if not 'variant' in preproc]

        # make sure the lists are the same length
        # pray to god that they are in the same order?
        # ^they appear to be in the same order
        length = len(subject_data['preproc'])
        print('preproc:{}'.format(str(length)))
        print('confounds:{}'.format(str(len(subject_data['confounds']))))
        print('brainmask:{}'.format(str(len(subject_data['brainmask']))))
        print('AROMAnoiseICs:{}'.format(str(len(subject_data['AROMAnoiseICs']))))
        print('MELODICmix:{}'.format(str(len(subject_data['MELODICmix']))))
        if any(len(lst) != length for lst in [subject_data['brainmask'],
                                              subject_data['confounds'],
                                              subject_data['AROMAnoiseICs'],
                                              subject_data['MELODICmix']]):
            raise ValueError('input lists are not the same length!')

        # if there are multiples, check to see what changes (session, task, run)

        # iterables
        # task [all]
        # run [all]
        # session [all]
        # space [not all], do not include
        single_subject_wf = init_single_subject_wf(AROMAnoiseICs=subject_data['AROMAnoiseICs'],
                                                   brainmask=subject_data['brainmask'],
                                                   confounds=subject_data['confounds'],
                                                   confound_names=confound_names,
                                                   deriv_pipe_dir=deriv_pipe_dir,
                                                   low_pass=low_pass,
                                                   MELODICmix=subject_data['MELODICmix'],
                                                   name='single_subject' + subject_id + '_wf'
                                                   preproc=subject_data['preproc'],
                                                   regfilt=regfilt,
                                                   res=res,
                                                   result_dir=result_dir,
                                                   run_id=run_id,
                                                   run_uuid=run_uuid,
                                                   ses_id=ses_id,
                                                   smooth=smooth,
                                                   space=space,
                                                   subject_id=subject_id,
                                                   task_id=task_id,
                                                   variant=variant)

        single_subject_wf.config['execution']['crashdump_dir'] = (
            os.path.join(result_dir, "sub-" + subject_id, 'log', run_uuid)
        )
        # single_subject_wf.base_dir = os.path.join(work_dir,
        #                                          'nuisance_regression_work', 'sub-'+subject_id)
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        nuisance_regression_wf.add_nodes([single_subject_wf])
    return nuisance_regression_wf


def init_single_subject_wf(subject_id, name, ses_id, result_dir, deriv_pipe_dir,
                           confound_names, confounds, task_id, space, variant, res,
                           run_uuid, smooth, low_pass, regfilt, run_id, preproc, brainmask,
                           AROMAnoiseICs, MELODICmix):
    r"""
    This workflow organizes the execution of a single subject.

    Parameters

        confound_names : list of str or None
            Column names from FMRIPREP's confounds tsv that were selected for
            nuisance regression
        deriv_pipe_dir : str
            The absolute path to the FMRIPREP directory
        low_pass : float or None
            The frequency to set low pass filter (in Hz)
        subject_id : str
            subject id to analyze
        work_dir : str
            The absolute path to execute workflows/nodes
        result_dir : str
            The absolute path to the base directory where results will be stored
        ses_id : str
            Session id to analyze
        task_id : str
            Task id to analyze
        space : str
            Output Space from FMRIPREP to analyze
        variant : str
            Output variant from FMRIPREP to analyze
        res : str
            Output resolution from FMRIPREP to analyze
        smooth : float or None
            smoothing kernel to apply to image (mm)
        run_id : str
            run number to analyze
        regfilt : bool
            Selects to run FilterRegressor from the output from ICA-AROMA
        run_uuid : str
            Unique identifier for execution instance
        AROMAnoiseICs : str
            Absolute path to csv file indicating noise ICs
        MELODICmix : str
            Absolute path to tsv listing all ICs
    """

    single_subject_wf = pe.Workflow(name=name)
    # import pdb; pdb.set_trace()
    # tmp = zip(preproc, brainmask, confounds, MELODICmix, AROMAnoiseICs)
    # print(str(tmp))
    inputnode = pe.Node(IdentityInterface(
        fields=['bold_preproc', 'bold_mask', 'confounds', 'MELODICmix', 'AROMAnoiseICs']),
        name='inputnode', synchronize=True,
        iterables=[('bold_preproc', 'bold_mask', 'confounds', 'MELODICmix', 'AROMAnoiseICs'),
                   zip(preproc, brainmask, confounds, MELODICmix, AROMAnoiseICs)])

    # inputnode.inputs.bold_preproc = preproc
    # inputnode.inputs.bold_mask = brainmask
    # inputnode.inputs.confounds = confounds
    # inputnode.inputs.MELODICmix = MELODICmix
    # inputnode.inputs.AROMAnoiseICs = AROMAnoiseICs

    outputnode = pe.Node(IdentityInterface(
        fields=['bold_resid']),
        name='outputnode')

    # datasink = pe.Node(nio.DataSink(), name='datasink')
    # subject_outdir = "sub-{}.ses-{}.func".format(subject_id, ses_id)
    # datasink.inputs.base_directory = result_dir

    # below obsolete, changed to iterables
    # Set input nodes
    # inputnode.inputs.bold_preproc = preproc
    # inputnode.inputs.bold_mask = brainmask
    # inputnode.inputs.confounds = confounds
    # inputnode.inputs.MELODICmix = MELODICmix
    # inputnode.inputs.AROMAnoiseICs = AROMAnoiseICs

    # workhorse workflow
    derive_residuals_wf = init_derive_residuals_wf(smooth=smooth,
                                                   confound_names=confound_names,
                                                   regfilt=regfilt,
                                                   lp=low_pass)

    derivatives_wf = init_derivatives_wf(result_dir=result_dir)
    # change name of resid.nii.gz
    # rename = pe.Node(Rename(format_string="sub-%(subject_id)s_ses-%(ses_id)s_task-%(task_id)s_bold_clean.nii.gz"),
    #                 name='rename')
    # rename.inputs.subject_id = subject_id
    # rename.inputs.task_id = task_id
    # rename.inputs.ses_id = ses_id

    single_subject_wf.connect([
        (inputnode, derive_residuals_wf, [('bold_preproc', 'inputnode.bold_preproc'),
                                          ('bold_mask', 'inputnode.bold_mask'),
                                          ('confounds', 'inputnode.confounds'),
                                          ('MELODICmix', 'inputnode.MELODICmix'),
                                          ('AROMAnoiseICs', 'inputnode.AROMAnoiseICs')]),
        (derive_residuals_wf, outputnode, [('outputnode.bold_resid', 'bold_resid')]),
        (inputnode, derivatives_wf, [('bold_preproc', 'inputnode.source_file')]),
        (outputnode, derivatives_wf, [('bold_resid', 'inputnode.bold_clean')]),
    ])

    return single_subject_wf


def init_derivatives_wf(result_dir, name='nuisance_regression_wf'):
    """
    Set up a battery of datasinks to store derivatives in the right location
    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(IdentityInterface(fields=['source_file', 'bold_clean']),
        name='inputnode')

    ds_bold_clean = pe.Node(DerivativesDataSink(base_directory=os.path.dirname(result_dir),
                                                suffix='clean',
                                                out_path_base=os.path.basename(result_dir)),
                            name='ds_bold_clean')

    workflow.connect([
    (inputnode, ds_bold_clean, [('source_file', 'source_file'),
                                ('bold_clean', 'in_file')])
    ])

    return workflow


"""
################################################################
################################################################
################################################################
"""

"""
################################################################
##################   NuisanceRegression   ######################
################################################################
"""


def init_derive_residuals_wf(name='derive_residuals_wf', t_r=2.0,
                             smooth=None, confound_names=None,
                             regfilt=False, lp=None):
    r"""
    This workflow derives the residual image from the preprocessed FMRIPREP image.

    Parameters

        name : str
            name of the workflow
        t_r : float
            time of repetition to collect a volume
        smooth : float or None
            smoothing kernel to apply to image (mm)
        confound_names : list of str or None
            Column names from FMRIPREP's confounds tsv that were selected for
            nuisance regression
        regfilt : bool
            Selects to run FilterRegressor from the output from ICA-AROMA
        lp : float or None
            The frequency to set low pass filter (in Hz)
    """
    # Steps
    # 1) brain mask
    # 2) smooth (optional)
    # 3) regfilt (optional)
    # 4) remove residuals

    inputnode = pe.Node(IdentityInterface(
        fields=['bold_preproc', 'bold_mask', 'confounds', 'MELODICmix',
                'AROMAnoiseICs']),
        name='inputnode')

    outputnode = pe.Node(IdentityInterface(
        fields=['bold_resid']),
        name='outputnode')

    # Function to perform confound removal
    def remove_confounds(nii, confounds, t_r=2.0, confound_names=None, lp=None):
        import nibabel as nib
        import pandas as pd
        import os
        from nilearn.image import clean_img
        img = nib.load(nii)
        confounds_pd = pd.read_csv(confounds, sep="\t")
        if confound_names is None:
            confound_names = [col for col in confounds_pd.columns
                              if 'CompCor' in col or 'X' in col or 'Y' in col or 'Z' in col]
        confounds_np = confounds_pd.as_matrix(columns=confound_names)
        kwargs = {
                  'imgs': img,
                  'confounds': confounds_np,
                  't_r': t_r
                 }
        if lp:
            kwargs['low_pass'] = lp
        cleaned_img = clean_img(**kwargs)
        working_dir = os.getcwd()
        resid_nii = os.path.join(working_dir, 'resid.nii.gz')
        nib.save(cleaned_img, resid_nii)

        return resid_nii

    # brain mask node
    mask_bold = pe.Node(MultiImageMaths(op_string='-mul %s'), name='mask_bold')
    # optional smoothing workflow
    smooth_wf = init_smooth_wf(smooth=smooth)
    # optional filtering workflow
    filt_reg_wf = init_filt_reg_wf(regfilt=regfilt)
    # residual node
    calc_resid = pe.Node(name='calc_resid',
                         interface=Function(input_names=['nii',
                                                         'confounds',
                                                         't_r',
                                                         'confound_names',
                                                         'lp'],
                                            output_names=['nii_resid'],
                                            function=remove_confounds))

    # Predefined attributes
    calc_resid.inputs.t_r = t_r
    calc_resid.inputs.confound_names = confound_names
    calc_resid.inputs.lp = lp

    # main workflow
    workflow = pe.Workflow(name=name)
    workflow.connect([
        (inputnode, mask_bold, [('bold_preproc', 'in_file'),
                                ('bold_mask', 'operand_files')]),
        (mask_bold, smooth_wf, [('out_file', 'inputnode.bold')]),
        (inputnode, smooth_wf, [('bold_mask', 'inputnode.bold_mask')]),
        (smooth_wf, filt_reg_wf, [('outputnode.bold_smooth', 'inputnode.bold')]),
        (inputnode, filt_reg_wf, [('bold_mask', 'inputnode.bold_mask'),
                                  ('MELODICmix', 'inputnode.MELODICmix'),
                                  ('AROMAnoiseICs', 'inputnode.AROMAnoiseICs')]),
        (filt_reg_wf, calc_resid, [('outputnode.bold_regfilt', 'nii')]),
        (inputnode, calc_resid, [('confounds', 'confounds')]),
        (calc_resid, outputnode, [('nii_resid', 'bold_resid')]),
    ])

    return workflow


# fsl regfilt workflow
def init_filt_reg_wf(name='filt_reg_wf', regfilt=None):
    inputnode = pe.Node(IdentityInterface(
        fields=['bold', 'bold_mask', 'MELODICmix', 'AROMAnoiseICs']),
        name='inputnode')

    outputnode = pe.Node(IdentityInterface(
        fields=['bold_regfilt']),
        name='outputnode')

    workflow = pe.Workflow(name=name)
    if regfilt:
        def csv_to_list(csv_f):
            import csv
            with open(csv_f) as f:
                reader = csv.reader(f, delimiter=str(','))
                mlist = list(reader)[0]
            return [int(x) for x in mlist]

        filter_regressor = pe.Node(FilterRegressor(), name='filter_regressor')
        workflow.connect([
            (inputnode, filter_regressor, [('bold', 'in_file'),
                                           ('bold_mask', 'mask'),
                                           ('MELODICmix', 'design_file'),
                                           (('AROMAnoiseICs', csv_to_list), 'filter_columns')]),
            (filter_regressor, outputnode, [('out_file', 'bold_regfilt')]),
        ])
    else:
        workflow.connect([
            (inputnode, outputnode, [('bold', 'bold_regfilt')]),
        ])

    return workflow


# smoothing workflow
def init_smooth_wf(name='smooth_wf', smooth=None):
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(IdentityInterface(
        fields=['bold', 'bold_mask']),
        name='inputnode')

    outputnode = pe.Node(IdentityInterface(
        fields=['bold_smooth']),
        name='outputnode')

    if smooth:
        calc_median_val = pe.Node(ImageStats(op_string='-k %s -p 50'), name='calc_median_val')
        calc_bold_mean = pe.Node(MeanImage(), name='calc_bold_mean')

        def getusans_func(image, thresh):
            return [tuple([image, thresh])]

        def _getbtthresh(medianval):
            return 0.75 * medianval
        getusans = pe.Node(Function(function=getusans_func, output_names=['usans']),
                           name='getusans', mem_gb=0.01)

        smooth = pe.Node(SUSAN(fwhm=smooth), name='smooth')

        workflow.connect([
            (inputnode, calc_median_val, [('bold', 'in_file'),
                                          ('bold_mask', 'mask_file')]),
            (inputnode, calc_bold_mean, [('bold', 'in_file')]),
            (calc_bold_mean, getusans, [('out_file', 'image')]),
            (calc_median_val, getusans, [('out_stat', 'thresh')]),
            (inputnode, smooth, [('bold', 'in_file')]),
            (getusans, smooth, [('usans', 'usans')]),
            (calc_median_val, smooth, [(('out_stat', _getbtthresh), 'brightness_threshold')]),
            (smooth, outputnode, [('smoothed_file', 'bold_smooth')]),
        ])
    else:
        workflow.connect([
            (inputnode, outputnode, [('bold', 'bold_smooth')]),
        ])

    return workflow


"""
################################################################
################################################################
################################################################
"""

"""
#################################################################
#################   USER INTERFACE (CLI)   ######################
#################################################################
"""


def get_parser():
    """Build parser object"""
    parser = argparse.ArgumentParser(description='NuisanceRegression BIDS arguments')
    parser.add_argument('deriv_pipe_dir', help='FMRIPREP directory')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('--participant_label', help='The label(s) of the participant(s) '
                        'that should be analyzed. The label '
                        'corresponds to sub-<participant_label> from the BIDS spec '
                        '(so it does not include "sub-"). If this parameter is not '
                        'provided all subjects should be analyzed. Multiple '
                        'participants can be specified with a space separated list.',
                        nargs="+")
    parser.add_argument('-w', '--work_dir', help='Directory where all intermediate '
                        'files are stored')
    # preprocessing options
    proc_opts = parser.add_argument_group('Options for preprocessing')
    proc_opts.add_argument('-sm', '--smooth', action='store', type=float,
                           help='select a smoothing kernel (mm)')
    proc_opts.add_argument('-l', '--low_pass', action='store', type=float,
                           default=None, help='low pass filter')
    proc_opts.add_argument('-f', '--regfilt', action='store_true', default=False,
                           help='Do non-aggressive filtering from ICA-AROMA')
    proc_opts.add_argument('-c', '--confounds', help='The confound column names '
                           'that are to be included in nuisance regression',
                           nargs="+")
    # Image Selection options
    image_opts = parser.add_argument_group('Options for selecting images')
    image_opts.add_argument('-t', '--task_id', action='store',
                            default=None, help='select a specific task to be processed')
    image_opts.add_argument('-sp', '--space', action='store',
                            default=None, help='select a bold derivative in a specific space to be used')
    image_opts.add_argument('--variant', action='store',
                            default=None, help='select a variant bold to process')
    image_opts.add_argument('--exclude_variant', action='store_true',
                            default=False, help='exclude the variant from FMRIPREP')
    image_opts.add_argument('-r', '--res', action='store',
                            default=None, help='select a resolution to analyze')
    image_opts.add_argument('--run', action='store',
                            default=None, help='select a run to analyze')
    image_opts.add_argument('--ses', action='store',
                            default=None, help='select a session to analyze')

    # misc options
    misc_opts = parser.add_argument_group('Options for miscellaneous abilities')
    misc_opts.add_argument('--graph', action='store_true', default=False,
                           help='generates a graph png of the workflow')
    return parser


def main():
    # Get commandline arguments
    opts = get_parser().parse_args()

    # get process options to name output directory
    if opts.smooth:
        smooth_name = '_{}mm'.format(opts.smooth)
    else:
        smooth_name = ''

    if opts.low_pass:
        lp_name = '_{}Hz'.format(opts.low_pass)
    else:
        lp_name = ''

    if opts.regfilt:
        regfilt_name = '_regfilt'
    else:
        regfilt_name = ''

    # Set up main directories
    output_dir = os.path.abspath(opts.output_dir)
    result_dir = os.path.join(output_dir,
                    'NuisanceRegression{}{}{}'.format(smooth_name, lp_name, regfilt_name)
                 )
    log_dir = os.path.join(result_dir, 'logs')
    work_dir = os.path.abspath(opts.work_dir)

    # make directories
    # Check and create output and working directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # Set up some instrumental utilities
    run_uuid = strftime('%Y%m%d-%H%M%S_') + str(uuid.uuid4())

    # Nipype config (logs and execution)
    ncfg.update_config({
        'logging': {'log_directory': log_dir,
                    'log_to_file': True},
        'execution': {'crashdump_dir': log_dir,
                      'crashfile_format': 'txt',
                      'parameterize_dirs': False,
                     },
    })

    # Nipype plugin configuration
    plugin_settings = {'plugin': 'Linear'}

    # only for a subset of subjects
    if opts.participant_label:
        subject_list = opts.participant_label
    # for all subjects
    else:
        subject_dirs = glob(os.path.join(opts.deriv_pipe_dir, "sub-*"))
        subject_list = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]

    nuisance_regression_wf = init_nuisance_regression_wf(
        confound_names=opts.confounds,
        deriv_pipe_dir=opts.deriv_pipe_dir,
        exclude_variant=opts.exclude_variant,
        low_pass=opts.low_pass,
        regfilt=opts.regfilt,
        res=opts.res,
        result_dir=result_dir,
        run_id=opts.run,
        run_uuid=run_uuid,
        ses_id=opts.ses,
        smooth=opts.smooth,
        space=opts.space,
        subject_list=subject_list,
        task_id=opts.task_id,
        variant=opts.variant,
        work_dir=work_dir,
    )
    if opts.graph:
        nuisance_regression_wf.write_graph(graph2use='colored',
            dotfilename=os.path.join(work_dir, 'graph_colored.dot'))

    try:
        nuisance_regression_wf.run(**plugin_settings)
    except RuntimeError as e:
        print('ERROR!')
        raise(e)


"""
################################################################
################################################################
################################################################
"""

# Let's you run the script directly (as opposed to importing it)
if __name__ == '__main__':
    main()
