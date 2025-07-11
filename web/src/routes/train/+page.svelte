<script lang="ts">
  import { onDestroy } from 'svelte';
  import TrainForm from '$lib/components/TrainForm.svelte';
  import TrainProgress from '$lib/components/TrainProgress.svelte';
  import { fetchTrainStatus, startTrain } from '$lib/api';
  import type { TrainRequest, TrainStatus } from '$lib/api';

  let status: TrainStatus | null = null;
  let jobId = '';
  let history: Array<{ epoch: number; loss: number; accuracy: number }> = [];
  let pollTimer: ReturnType<typeof setInterval> | null = null;

  function stopPolling() {
    if (pollTimer !== null) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function recordEpoch(s: TrainStatus) {
    if (s.loss != null && s.accuracy != null && s.epoch >= history.length) {
      history = [...history, { epoch: s.epoch, loss: s.loss, accuracy: s.accuracy }];
    }
  }

  async function poll() {
    if (!jobId) return;
    try {
      status = await fetchTrainStatus(jobId);
      recordEpoch(status);
      if (status.status !== 'running') stopPolling();
    } catch {
      stopPolling();
    }
  }

  async function start(request: TrainRequest) {
    stopPolling();
    history = [];
    const job = await startTrain(request);
    jobId = job.job_id;
    status = await fetchTrainStatus(jobId);
    if (status.status === 'running') {
      pollTimer = setInterval(poll, 1000);
    }
  }

  onDestroy(stopPolling);
</script>

<div class="grid">
  <TrainForm backend="gpu" onStart={start} />
  <TrainProgress {status} {history} />
</div>
