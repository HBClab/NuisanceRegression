#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
'''
Workflow for doing preprocessing
that FMRIPREP doesn't complete, and derives standardized residuals from bold.
'''

from __future__ import print_function, division, absolute_import, unicode_literals
import argparse
import os
import pdb
from glob import glob
import uuid
# from niworkflows.nipype import config, logging
# config.enable_debug_mode()
# logging.update_logging(config)
from copy import deepcopy
from time import strftime
from bids.grabbids import BIDSLayout
import niworkflows.nipype.interfaces.io as nio
import niworkflows.nipype.pipeline.engine as pe
from niworkflows.nipype import config as ncfg
from niworkflows.nipype.interfaces.utility import IdentityInterface
from niworkflows.nipype.interfaces.fsl import ImageStats, MultiImageMaths, SUSAN
from niworkflows.nipype.interfaces.fsl.utils import FilterRegressor
from niworkflows.nipype.interfaces.fsl.maths import MeanImage
from niworkflows.nipype.interfaces.utility import Function
from nilearn.image import clean_img
import nibabel as nib
import pandas as pd
################################################################
#########################   GLOBAL   ###########################
################################################################

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

################################################################
################################################################
################################################################

################################################################
#######################  SETUP/RUN   ###########################
################################################################
def init_nuisance_regression_wf(confound_names, deriv_pipe_dir, low_pass,
                                subject_list, work_dir, result_dir,
                                ses_id, task_id, space, variant, res,
                                smooth, run_id,  regfilt, run_uuid):
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
    nuisance_regression_wf.base_dir = work_dir
    # get a representation of the directory/data structure
    data_layout = BIDSLayout(deriv_pipe_dir, bids_deriv_config)

    for subject_id in subject_list:
        # get preproc img
        preproc_query = {
                         'subject': subject_id,
                         'task': task_id,
                         'type': 'preproc',
                         'return_type': 'file',
                         'extensions': ['nii', 'nii.gz']
                        }
        if ses_id:
            preproc_query['session'] = ses_id
        if variant:
            preproc_query['variant'] = variant
        if space:
            preproc_query['space'] = space
        if res:
            preproc_query['res'] = res
        if run_id:
            preproc_query['run'] = run_id

        preproc_list = data_layout.get(**preproc_query)

        if not preproc_list:
            raise Exception("No preproc files were found for participant {}".format(subject_id))
        elif len(preproc_list) > 1:
            raise Exception("Too many preproc files were found for participant {}".format(subject_id))
        else:
            preproc_file = preproc_list[0]

        # brainmask
        brainmask_query = {
                           'subject': subject_id,
                           'task': task_id,
                           'type': 'brainmask',
                           'return_type': 'file',
                           'extensions': ['nii', 'nii.gz']
                          }
        if ses_id:
            brainmask_query['session'] = ses_id
        if space:
            brainmask_query['space'] = space
        if res:
            brainmask_query['res'] = res
        if run_id:
            brainmask_query['run'] = run_id

        brainmask_list = data_layout.get(**brainmask_query)

        if not brainmask_list:
            raise Exception("No brainmask files were found for participant {}".format(subject_id))
        elif len(brainmask_list) > 1:
            raise Exception("Too many brainmask files were found for participant {}".format(subject_id))
        else:
            brainmask_file = brainmask_list[0]

        # confounds
        confounds_query = {
                           'subject': subject_id,
                           'task': task_id,
                           'type': 'confounds',
                           'extensions': 'tsv',
                           'return_type': 'file'
                          }
        if ses_id:
            confounds_query['session'] = ses_id
        if run_id:
            confounds_query['run'] = run_id

        confounds_list = data_layout.get(**confounds_query)

        if not confounds_list:
            raise Exception("No confound files were found for participant {}".format(subject_id))
        elif len(confounds_list) > 1:
            raise Exception("Too many confound files were found for participant {}".format(subject_id))
        else:
            confounds_file = confounds_list[0]

        # confounds
        MELODICmix_query = {
                           'subject': subject_id,
                           'task': task_id,
                           'type': 'MELODICmix',
                           'extensions': 'tsv',
                           'return_type': 'file'
                          }
        if ses_id:
            MELODICmix_query['session'] = ses_id
        if run_id:
            MELODICmix_query['run'] = run_id

        MELODICmix_list = data_layout.get(**MELODICmix_query)

        if not MELODICmix_list:
            raise Exception("No MELODICmix files were found for participant {}".format(subject_id))
        elif len(MELODICmix_list) > 1:
            raise Exception("Too many MELODICmix files were found for participant {}".format(subject_id))
        else:
            MELODICmix_file = MELODICmix_list[0]

        # confounds
        AROMAnoiseICs_query = {
                           'subject': subject_id,
                           'task': task_id,
                           'type': 'AROMAnoiseICs',
                           'extensions': 'csv',
                           'return_type': 'file'
                          }
        if ses_id:
            AROMAnoiseICs_query['session'] = ses_id
        if run_id:
            AROMAnoiseICs_query['run'] = run_id

        AROMAnoiseICs_list = data_layout.get(**AROMAnoiseICs_query)

        if not AROMAnoiseICs_list:
            raise Exception("No AROMAnoiseICs files were found for participant {}".format(subject_id))
        elif len(AROMAnoiseICs_list) > 1:
            raise Exception("Too many AROMAnoiseICs files were found for participant {}".format(subject_id))
        else:
            AROMAnoiseICs_file = AROMAnoiseICs_list[0]

        single_subject_wf = init_single_subject_wf(AROMAnoiseICs=AROMAnoiseICs_file,
                                                   brainmask=brainmask_file,
                                                   confounds=confounds_file,
                                                   confound_names=confound_names,
                                                   deriv_pipe_dir=deriv_pipe_dir,
                                                   low_pass=low_pass,
                                                   MELODICmix=MELODICmix_file,
                                                   preproc=preproc_file,
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
                                                   variant=variant
        )

        single_subject_wf.config['execution']['crashdump_dir'] = (
            os.path.join(result_dir, "sub-" + subject_id, 'log', run_uuid)
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        nuisance_regression_wf.add_nodes([single_subject_wf])
    return nuisance_regression_wf


def init_single_subject_wf(subject_id, ses_id, result_dir, deriv_pipe_dir,
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

    single_subject_wf = pe.Workflow(name='single_subject_wf')
    inputnode = pe.Node(IdentityInterface(
        fields=['bold_preproc', 'bold_mask', 'confounds', 'MELODICmix', 'AROMAnoiseICs']),
        name='inputnode')
    outputnode = pe.Node(IdentityInterface(
        fields=['bold_resid']),
        name='outputnode')

    datasink = pe.Node(nio.DataSink(), name='datasink')
    subject_outdir = "sub-{}.ses-{}.func.task-{}".format(subject_id, ses_id, task_id)
    datasink.inputs.base_directory = result_dir

    # Set input nodes
    inputnode.inputs.bold_preproc = preproc
    inputnode.inputs.bold_mask = brainmask
    inputnode.inputs.confounds = confounds
    inputnode.inputs.MELODICmix = MELODICmix
    inputnode.inputs.AROMAnoiseICs = AROMAnoiseICs

    derive_residuals_wf = init_derive_residuals_wf(smooth=smooth,
                                                   confound_names=confound_names,
                                                   regfilt=regfilt,
                                                   lp=low_pass)
    single_subject_wf.connect([
        (inputnode, derive_residuals_wf, [('bold_preproc', 'inputnode.bold_preproc'),
                                          ('bold_mask', 'inputnode.bold_mask'),
                                          ('confounds', 'inputnode.confounds'),
                                          ('MELODICmix', 'inputnode.MELODICmix'),
                                          ('AROMAnoiseICs', 'inputnode.AROMAnoiseICs')]),
        (derive_residuals_wf, outputnode, [('outputnode.bold_resid', 'bold_resid')]),
        (outputnode, datasink, [('bold_resid', subject_outdir)]),
    ])

    return single_subject_wf

################################################################
################################################################
################################################################

################################################################
############   PREPROCESSING/NUISANCE REGRESSION   #############
################################################################


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

################################################################
################################################################
################################################################


#################################################################
#####################   USER INTERFACE   ########################
#################################################################
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
                           help='low pass filter')
    proc_opts.add_argument('-f', '--regfilt', action='store_true', default=False,
                           help='Do non-aggressive filtering from ICA-AROMA')
    proc_opts.add_argument('-c', '--confounds', help='The confound column names '
                           'that are to be included in nuisance regression')
    # Image Selection options
    image_opts = parser.add_argument_group('Options for selecting images')
    image_opts.add_argument('-t', '--task_id', action='store',
                            help='select a specific task to be processed')
    image_opts.add_argument('-sp', '--space', action='store',
                            help='select a bold derivative in a specific space to be used')
    image_opts.add_argument('--variant', action='store',
                            help='select a variant bold to process')
    image_opts.add_argument('-r', '--res', action='store',
                            help='select a resolution to analyze')
    image_opts.add_argument('--run', action='store',
                            help='select a run to analyze')
    image_opts.add_argument('--ses', action='store', help='select a session to analyze')

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
        'NuisanceRegression{}{}{}'.format(smooth_name, lp_name, regfilt_name))
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
            dotfilename=os.path.join(work_dir,'graph_colored.dot'))

    try:
        nuisance_regression_wf.run(**plugin_settings)
    except RuntimeError as e:
        print('ERROR!')
        raise(e)


if __name__ == '__main__':
    main()
