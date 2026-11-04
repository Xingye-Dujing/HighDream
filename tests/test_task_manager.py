# pylint: disable=redefined-outer-name,missing-function-docstring
"""TaskManager worker-slot handling.

Regression: a task whose future timed out used to release its semaphore slot
immediately while the computation kept running, so hung work accumulated in the
executor and every later task queued behind it (and timed out too).
"""

import time

import routes.task_manager as tm


def _wait_status(task_id, statuses, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = tm.task_manager.get_task_status(task_id)
        if info is not None and info['status'] in statuses:
            return info
        time.sleep(0.02)
    raise AssertionError(f'task {task_id} never reached {statuses}')


def test_timed_out_tasks_hold_their_slot_until_finished(monkeypatch):
    sleep_seconds = 0.5
    monkeypatch.setattr(tm, 'start_compute',
                        lambda op, data: (time.sleep(sleep_seconds), (True, 'late'))[1])

    mgr = tm.task_manager
    slow_ids = [mgr.create_task('auto', {'expression': f'slow{i}'}, timeout_seconds=0.1)
                for i in range(tm.MAX_WORKERS)]
    for task_id in slow_ids:
        info = _wait_status(task_id, {'failed'})
        assert '超时' in info['error']

    # Every slot is still held by an abandoned computation.
    assert mgr._semaphore._value == 0  # pylint: disable=protected-access

    started = time.time()
    fast_id = mgr.create_task('auto', {'expression': 'fast'}, timeout_seconds=10)
    info = _wait_status(fast_id, {'completed'})
    assert info['result'] == 'late'
    # The fast task could only run after a leaked computation released a slot.
    assert time.time() - started >= sleep_seconds * 0.5


def test_task_that_cannot_get_a_slot_fails_cleanly(monkeypatch):
    monkeypatch.setattr(tm, 'start_compute',
                        lambda op, data: (time.sleep(1.0), (True, 'late'))[1])
    mgr = tm.task_manager
    blockers = [mgr.create_task('auto', {'expression': f'hold{i}'}, timeout_seconds=0.1)
                for i in range(tm.MAX_WORKERS)]
    time.sleep(0.3)

    queued = mgr.create_task('auto', {'expression': 'queued'}, timeout_seconds=0.2)
    info = _wait_status(queued, {'failed'})
    assert '排队' in info['error']

    for task_id in blockers:
        _wait_status(task_id, {'failed'})
