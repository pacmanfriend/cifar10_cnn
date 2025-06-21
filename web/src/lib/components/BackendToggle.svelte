<script lang="ts">
  import type { Backend } from '$lib/api';

  export let backend: Backend = 'cpu';
  export let disabled = false;
  export let onChange: (backend: Backend) => void = () => {};
</script>

<div class="actions">
  <button class:secondary={backend !== 'cpu'} disabled={disabled} on:click={() => onChange('cpu')}>
    CPU
  </button>
  <button class:secondary={backend !== 'gpu'} disabled={disabled} on:click={() => onChange('gpu')}>
    GPU
  </button>
</div>
