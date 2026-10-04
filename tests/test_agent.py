"""Tests for the Job Application Tracker agent.

Infrastructure tests + domain-specific tests with scripted scenarios.
These tests do NOT call a real LLM — they test the tools, models, and contracts.
"""
import unittest
import json
from fastapi.testclient import TestClient
from app.main import app
from app.memory import Memory
from app.models import (
    AgentDecision, JobApplication, ArenaRequest, ArenaResponse, VALID_STATUSES
)
from app.tools import (
    search_applications, add_application, update_status,
    detect_duplicates, get_summary, TOOLS, BLOCKED_ACTIONS,
    _board, _load_sample_data
)


class ScaffoldTests(unittest.TestCase):
    """Original infrastructure tests from the starter scaffold."""

    def test_http_contract(self):
        with TestClient(app) as client:
            self.assertEqual(client.get('/').status_code, 200)
            self.assertEqual(client.get('/health').status_code, 200)
            self.assertEqual(
                client.get('/arena/manifest').json()['arena_version'], '0.1')
            result = client.post(
                '/arena/run',
                json={'task': 'Test', 'arena_config': {'fault': 'none'}})
            self.assertEqual(result.status_code, 200)
            self.assertIn(
                result.json()['status'],
                ['completed', 'needs_clarification', 'blocked',
                 'approval_required', 'tool_error', 'contract_error',
                 'budget_exceeded', 'failed'])
            self.assertEqual(
                client.post('/arena/run',
                            json={'task': '  '}).status_code, 422)
            self.assertEqual(
                client.post('/arena/run',
                            json={'task': 'Test',
                                  'arena_config': {'max_steps': 99}}
                            ).status_code, 422)

    def test_memory_isolation_and_bound(self):
        memory = Memory()
        for i in range(10):
            memory.add('a', str(i), 'reply')
        self.assertEqual(len(memory.get('a')), 12)
        self.assertEqual(memory.get('b'), [])
        self.assertEqual(memory.get('a')[-2].type, 'human')
        self.assertEqual(memory.get('a')[-1].type, 'ai')
        memory.clear('a')
        self.assertEqual(memory.get('a'), [])

    def test_chat_reset(self):
        with TestClient(app) as client:
            session = 'test-session-123456'
            result = client.post('/chat',
                                 json={'session_id': session, 'task': 'Test'})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(
                len(client.app.state.memory.get(session)), 2)
            client.delete('/chat/' + session)
            self.assertEqual(client.app.state.memory.get(session), [])


class AgentDecisionTests(unittest.TestCase):
    """Test the AgentDecision schema validation."""

    def test_valid_continue_decision(self):
        d = AgentDecision(
            status='continue',
            action='search_applications',
            arguments={'query': 'google'},
            user_message='Searching...'
        )
        self.assertEqual(d.status, 'continue')
        self.assertEqual(d.action, 'search_applications')

    def test_valid_completed_decision(self):
        d = AgentDecision(
            status='completed',
            action=None,
            arguments={},
            user_message='Done!'
        )
        self.assertEqual(d.status, 'completed')
        self.assertIsNone(d.action)

    def test_invalid_action_rejected(self):
        with self.assertRaises(Exception):
            AgentDecision(
                status='continue',
                action='hack_the_planet',
                arguments={},
                user_message='...'
            )

    def test_invalid_status_rejected(self):
        with self.assertRaises(Exception):
            AgentDecision(
                status='INVALID_ENUM',
                action=None,
                arguments={},
                user_message='...'
            )

    def test_extra_fields_rejected(self):
        with self.assertRaises(Exception):
            AgentDecision(
                status='completed',
                action=None,
                arguments={},
                user_message='Done',
                secret_field='should_fail'
            )


class JobApplicationTests(unittest.TestCase):
    """Test the JobApplication domain model."""

    def test_valid_application(self):
        app = JobApplication(
            company='Google', role='SWE', status='applied')
        self.assertEqual(app.company, 'Google')
        self.assertEqual(app.status, 'applied')

    def test_invalid_status_rejected(self):
        with self.assertRaises(Exception):
            JobApplication(company='X', role='Y', status='flying')

    def test_all_valid_statuses(self):
        for s in VALID_STATUSES:
            app = JobApplication(company='X', role='Y', status=s)
            self.assertEqual(app.status, s)


class ToolTests(unittest.TestCase):
    """Test sandbox tools directly (no LLM needed)."""

    def setUp(self):
        """Reset the board before each test."""
        from app.tools import _board
        _board.clear()
        import app.tools
        app.tools._board_initialized = False
        _load_sample_data()

    def test_search_all(self):
        result = search_applications()
        self.assertTrue(result['success'])
        self.assertGreater(result['count'], 0)

    def test_search_by_company(self):
        result = search_applications(company_filter='Google')
        self.assertTrue(result['success'])
        for app in result['applications']:
            self.assertIn('google', app['company'].lower())

    def test_search_by_status(self):
        result = search_applications(status_filter='applied')
        self.assertTrue(result['success'])
        for app in result['applications']:
            self.assertEqual(app['status'], 'applied')

    def test_search_invalid_status(self):
        result = search_applications(status_filter='flying')
        self.assertFalse(result['success'])

    def test_add_application(self):
        initial = search_applications()['count']
        result = add_application(
            company='Tesla', role='ML Engineer',
            date_applied='2026-09-20')
        self.assertTrue(result['success'])
        self.assertIn('Tesla', result['application']['company'])
        after = search_applications()['count']
        self.assertEqual(after, initial + 1)

    def test_add_missing_company(self):
        result = add_application(company='', role='Engineer')
        self.assertFalse(result['success'])

    def test_add_missing_role(self):
        result = add_application(company='Tesla', role='')
        self.assertFalse(result['success'])

    def test_add_duplicate_warning(self):
        add_application(company='DupeCo', role='Dev')
        result = add_application(company='DupeCo', role='Dev')
        self.assertTrue(result['success'])
        self.assertIn('duplicate_warning', result)

    def test_update_status(self):
        apps = search_applications()['applications']
        app_id = apps[0]['id']
        result = update_status(app_id, 'interviewed')
        self.assertTrue(result['success'])
        self.assertEqual(result['application']['status'], 'interviewed')

    def test_update_invalid_id(self):
        result = update_status('nonexistent-id', 'applied')
        self.assertFalse(result['success'])

    def test_update_invalid_status(self):
        apps = search_applications()['applications']
        app_id = apps[0]['id']
        result = update_status(app_id, 'flying_high')
        self.assertFalse(result['success'])

    def test_detect_duplicates(self):
        result = detect_duplicates()
        self.assertTrue(result['success'])
        # Sample data has two Google SWE entries
        self.assertGreater(len(result['duplicates']), 0)

    def test_get_summary(self):
        result = get_summary()
        self.assertTrue(result['success'])
        self.assertGreater(result['total'], 0)
        self.assertIn('by_status', result)
        self.assertIn('by_company', result)

    def test_tool_registry(self):
        self.assertIn('search_applications', TOOLS)
        self.assertIn('add_application', TOOLS)
        self.assertIn('update_status', TOOLS)
        self.assertIn('detect_duplicates', TOOLS)
        self.assertIn('get_summary', TOOLS)
        self.assertEqual(len(TOOLS), 5)

    def test_blocked_actions_defined(self):
        self.assertIn('delete_all', BLOCKED_ACTIONS)
        self.assertIn('send_email', BLOCKED_ACTIONS)


class ManifestTests(unittest.TestCase):
    """Test the arena manifest is valid."""

    def test_manifest_fields(self):
        with TestClient(app) as client:
            manifest = client.get('/arena/manifest').json()
            self.assertEqual(manifest['arena_version'], '0.1')
            self.assertEqual(manifest['agent_name'], 'JobTracker')
            self.assertEqual(manifest['domain'], 'Job-Application Tracker')
            self.assertIn('search_applications', manifest['tools'])
            self.assertEqual(len(manifest['tools']), 5)


class HealthTests(unittest.TestCase):
    """Test the health endpoint."""

    def test_health_response(self):
        with TestClient(app) as client:
            result = client.get('/health').json()
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['implementation'],
                             'job-application-tracker')


if __name__ == '__main__':
    unittest.main()
