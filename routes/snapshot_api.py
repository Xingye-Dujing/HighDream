"""Notebook history snapshot endpoints."""

import json
import sqlite3

from flask import jsonify, request

from utils import snapshot_store

from .api_blueprint import api

_STORAGE_ERRORS = (sqlite3.Error, OSError)


@api.route('/snapshot_save', methods=['POST'])
def snapshot_save():
    """Store the current notebook as one history snapshot in the database.

    Expected JSON input:
        cells (list): notebook cell array (the page's ``getNotebookData()``)
        title (str, optional): display name; defaults to a timestamp

    Returns:
        JSON with success and the new snapshot metadata.
    """
    data = request.json or {}
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': '请求体必须是 JSON 对象'}), 400
    cells = data.get('cells')
    if not isinstance(cells, list) or not cells:
        return jsonify({'success': False, 'error': '笔记本为空'})
    try:
        title = str(data.get('title') or '').strip()
        meta = snapshot_store.create_snapshot(
            json.dumps(cells, ensure_ascii=False), title
        )
    except (ValueError, TypeError, AttributeError) as exc:
        return jsonify({'success': False, 'error': str(exc)})
    except _STORAGE_ERRORS:
        return jsonify({'success': False, 'error': '快照存储不可用，请检查缓存目录权限'})
    return jsonify({'success': True, 'snapshot': meta})


@api.route('/snapshot_list', methods=['GET'])
def snapshot_list():
    """Return recent snapshots (metadata only, newest first)."""
    try:
        max_rows = int(request.args.get('limit', '50'))
    except (TypeError, ValueError):
        max_rows = 50
    try:
        snapshots = snapshot_store.list_snapshots(max_rows)
    except _STORAGE_ERRORS:
        return jsonify({'success': False, 'error': '快照存储不可用，请检查缓存目录权限'})
    return jsonify({'success': True, 'snapshots': snapshots})


@api.route('/snapshot_get', methods=['GET'])
def snapshot_get():
    """Return one snapshot including its stored notebook payload."""
    try:
        snapshot_id = int(request.args.get('id', ''))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': '无效的快照编号'})
    try:
        snapshot = snapshot_store.get_snapshot(snapshot_id)
    except _STORAGE_ERRORS:
        return jsonify({'success': False, 'error': '快照存储不可用，请检查缓存目录权限'})
    if snapshot is None:
        return jsonify({'success': False, 'error': '快照不存在'})
    return jsonify({'success': True, 'snapshot': snapshot})


@api.route('/snapshot_delete', methods=['POST'])
def snapshot_delete():
    """Delete one snapshot by id."""
    data = request.json or {}
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': '请求体必须是 JSON 对象'}), 400
    try:
        snapshot_id = int(data.get('id'))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': '无效的快照编号'})
    try:
        removed = snapshot_store.delete_snapshot(snapshot_id)
    except _STORAGE_ERRORS:
        return jsonify({'success': False, 'error': '快照存储不可用，请检查缓存目录权限'})
    return jsonify({'success': removed, 'error': '' if removed else '快照不存在'})
