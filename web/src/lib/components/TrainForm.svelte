<script lang="ts">
  import type { Backend, TrainRequest } from '$lib/api';

  export let backend: Backend = 'cpu';
  export let onStart: (request: TrainRequest) => void = () => {};

  let data_dir = 'data/cifar-10-batches-bin';
  let epochs = 1;
  let batch_size = 64;
  let learning_rate = 0.003;
  let momentum = 0.9;
  let lr_decay_epochs = 5;
  let lr_decay_factor = 0.5;

  function submit() {
    onStart({
      data_dir,
      backend,
      epochs,
      batch_size,
      learning_rate,
      momentum,
      lr_decay_epochs,
      lr_decay_factor
    });
  }
</script>

<section class="panel">
  <h2>Train</h2>
  <div class="form">
    <label>Data directory<input bind:value={data_dir} /></label>
    <label>Epochs<input type="number" min="0" bind:value={epochs} /></label>
    <label>Batch size<input type="number" min="1" bind:value={batch_size} /></label>
    <label>Learning rate<input type="number" step="0.0001" bind:value={learning_rate} /></label>
    <label>Momentum<input type="number" step="0.05" min="0" max="1" bind:value={momentum} /></label>
    <label>Decay epochs<input type="number" min="0" bind:value={lr_decay_epochs} /></label>
    <label>Decay factor<input type="number" step="0.05" min="0.01" bind:value={lr_decay_factor} /></label>
  </div>
  <div class="actions">
    <button on:click={submit}>Start</button>
  </div>
</section>

<style>
  .form {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin-bottom: 14px;
  }
</style>
