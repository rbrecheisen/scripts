% jobfile = {'/Users/Ralph/datasets/imagemend/scripts/run_spm12_job.m'};
stopfile = '/data/raw_data/imagemend/uio/smri/stop.txt'
jobfile = {'/data/software/scripts/run_spm12_job.m'};
jobs = repmat(jobfile, 1, 1);
inputs = cell(0, 1);

spm('defaults', 'PET');
spm_jobman('initcfg');
spm_jobman('run', jobs, inputs{:});

f = fopen(stopfile, 'w');
fclose(f);
