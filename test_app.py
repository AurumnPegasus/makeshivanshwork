"""Tests for the Make Shivansh Work app - focusing on new reads feature and UI fixes."""

import pytest
import os

# Set up test environment before importing app
os.environ['DATABASE'] = ':memory:'

from app import app, init_db, get_db, execute_query, add_link


@pytest.fixture
def client():
    """Create test client with fresh database."""
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-secret'

    with app.test_client() as client:
        with app.app_context():
            # Reinitialize DB for each test
            conn = get_db()
            conn.executescript('''
                DROP TABLE IF EXISTS reads;
                DROP TABLE IF EXISTS tasks;
                DROP TABLE IF EXISTS users;
                DROP TABLE IF EXISTS chat_history;
                DROP TABLE IF EXISTS magic_links;
            ''')
            conn.commit()
            conn.close()
            init_db()
        yield client


@pytest.fixture
def auth_client(client):
    """Create authenticated test client."""
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['email'] = 'test@fydy.ai'

    # Create test user
    conn = get_db()
    execute_query(conn, 'INSERT INTO users (id, email) VALUES (?, ?)', (1, 'test@fydy.ai'))
    conn.commit()
    conn.close()

    return client


class TestReadsAPI:
    """Test the reads CRUD API endpoints."""

    def test_get_reads_empty(self, auth_client):
        """GET /api/reads returns empty list initially."""
        resp = auth_client.get('/api/reads')
        assert resp.status_code == 200
        assert resp.json == []

    def test_create_read(self, auth_client):
        """POST /api/reads creates a new read."""
        resp = auth_client.post('/api/reads',
            json={'title': 'Test Paper', 'url': 'https://arxiv.org/abs/1234', 'author': 'Test Author'},
            content_type='application/json')

        assert resp.status_code == 201
        data = resp.json
        assert data['title'] == 'Test Paper'
        assert data['url'] == 'https://arxiv.org/abs/1234'
        assert data['author'] == 'Test Author'
        assert data['status'] == 'unread'
        assert 'id' in data

    def test_create_read_title_required(self, auth_client):
        """POST /api/reads requires title."""
        resp = auth_client.post('/api/reads',
            json={'url': 'https://example.com'},
            content_type='application/json')

        assert resp.status_code == 400
        assert 'error' in resp.json

    def test_get_reads_with_data(self, auth_client):
        """GET /api/reads returns created reads."""
        # Create a read first
        auth_client.post('/api/reads',
            json={'title': 'Paper 1'},
            content_type='application/json')
        auth_client.post('/api/reads',
            json={'title': 'Paper 2', 'status': 'reading'},
            content_type='application/json')

        resp = auth_client.get('/api/reads')
        assert resp.status_code == 200
        assert len(resp.json) == 2

    def test_update_read(self, auth_client):
        """PUT /api/reads/<id> updates a read."""
        # Create a read
        create_resp = auth_client.post('/api/reads',
            json={'title': 'Original Title'},
            content_type='application/json')
        read_id = create_resp.json['id']

        # Update it
        resp = auth_client.put(f'/api/reads/{read_id}',
            json={'title': 'Updated Title', 'status': 'reading'},
            content_type='application/json')

        assert resp.status_code == 200
        assert resp.json['title'] == 'Updated Title'
        assert resp.json['status'] == 'reading'

    def test_update_read_not_found(self, auth_client):
        """PUT /api/reads/<id> returns 404 for non-existent read."""
        resp = auth_client.put('/api/reads/9999',
            json={'title': 'Test'},
            content_type='application/json')

        assert resp.status_code == 404

    def test_delete_read(self, auth_client):
        """DELETE /api/reads/<id> deletes a read."""
        # Create a read
        create_resp = auth_client.post('/api/reads',
            json={'title': 'To Delete'},
            content_type='application/json')
        read_id = create_resp.json['id']

        # Delete it
        resp = auth_client.delete(f'/api/reads/{read_id}')
        assert resp.status_code == 204

        # Verify it's gone
        get_resp = auth_client.get('/api/reads')
        assert len(get_resp.json) == 0

    def test_filter_reads_by_status(self, auth_client):
        """GET /api/reads?status=X filters by status."""
        # Create reads with different statuses
        auth_client.post('/api/reads', json={'title': 'Unread Paper'}, content_type='application/json')

        create_resp = auth_client.post('/api/reads', json={'title': 'Reading Paper'}, content_type='application/json')
        auth_client.put(f"/api/reads/{create_resp.json['id']}",
            json={'status': 'reading'}, content_type='application/json')

        # Filter by status
        resp = auth_client.get('/api/reads?status=reading')
        assert resp.status_code == 200
        assert len(resp.json) == 1
        assert resp.json[0]['title'] == 'Reading Paper'


class TestAddLink:
    """Test the add_link function."""

    def test_add_link_returns_result(self):
        """add_link returns URL and title."""
        result = add_link('A Path to Autonomous Machine Intelligence')

        # Should return a dict with url, title
        assert 'url' in result
        assert 'title' in result

    def test_add_link_handles_any_query(self):
        """add_link always returns a result (fallback to Google search)."""
        result = add_link('some random query xyz')
        # Always returns a result (worst case: Google link)
        assert 'url' in result
        assert 'title' in result


class TestDatabaseSchema:
    """Test database schema includes reads table."""

    def test_reads_table_exists(self, auth_client):
        """The reads table should exist with correct columns."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(reads)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        expected = {'id', 'title', 'url', 'author', 'notes', 'status', 'added_by', 'created_at', 'updated_at'}
        assert expected.issubset(columns)


class TestAuthRequired:
    """Test that API endpoints require authentication."""

    def test_reads_requires_auth(self, client):
        """Reads endpoints should redirect when not authenticated."""
        resp = client.get('/api/reads')
        assert resp.status_code == 302  # Redirect to login

    def test_tasks_requires_auth(self, client):
        """Tasks endpoints should redirect when not authenticated."""
        resp = client.get('/api/tasks')
        assert resp.status_code == 302


class TestChatRobustness:
    """Test chat endpoint handles edge cases gracefully."""

    def test_empty_response_with_action_shows_summary(self, auth_client):
        """When Gemini returns no text but performs action, show summary."""
        from unittest.mock import patch, MagicMock

        def create_mock_response(text='', function_calls=None):
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_content = MagicMock()
            parts = []
            if text:
                mock_text_part = MagicMock()
                mock_text_part.text = text
                mock_text_part.function_call = None
                parts.append(mock_text_part)
            if function_calls:
                for fc in function_calls:
                    mock_fc_part = MagicMock()
                    mock_fc_part.text = None
                    mock_fc_part.function_call = MagicMock()
                    mock_fc_part.function_call.name = fc['name']
                    mock_fc_part.function_call.args = fc.get('args', {})
                    parts.append(mock_fc_part)
            mock_content.parts = parts
            mock_candidate.content = mock_content
            mock_response.candidates = [mock_candidate]
            return mock_response

        with patch('app.gemini_client') as mock_gemini:
            mock_chat = MagicMock()
            # Function call with NO text
            first_response = create_mock_response(
                text='',
                function_calls=[{'name': 'add_read', 'args': {'title': 'Test Book'}}]
            )
            second_response = create_mock_response(text='')
            mock_chat.send_message.side_effect = [first_response, second_response]
            mock_gemini.chats.create.return_value = mock_chat

            resp = auth_client.post('/api/chat', json={'message': 'Add test book'})
            data = resp.get_json()

            assert resp.status_code == 200
            assert data.get('response'), 'Response should not be empty'
            assert 'Test Book' in data.get('response', '')

    def test_api_error_returns_500(self, auth_client):
        """API errors should return 500 with error message."""
        from unittest.mock import patch

        with patch('app.gemini_client') as mock_gemini:
            mock_gemini.chats.create.side_effect = Exception('API error')

            resp = auth_client.post('/api/chat', json={'message': 'Hello'})
            data = resp.get_json()

            assert resp.status_code == 500
            assert 'error' in data
            assert 'API error' in data['error']

    def test_normal_response_unchanged(self, auth_client):
        """Normal text responses should work as before."""
        from unittest.mock import patch, MagicMock

        with patch('app.gemini_client') as mock_gemini:
            mock_chat = MagicMock()
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_content = MagicMock()
            mock_text_part = MagicMock()
            mock_text_part.text = 'Hello! How can I help?'
            mock_text_part.function_call = None
            mock_content.parts = [mock_text_part]
            mock_candidate.content = mock_content
            mock_response.candidates = [mock_candidate]
            mock_chat.send_message.return_value = mock_response
            mock_gemini.chats.create.return_value = mock_chat

            resp = auth_client.post('/api/chat', json={'message': 'Hello'})
            data = resp.get_json()

            assert resp.status_code == 200
            assert data.get('response') == 'Hello! How can I help?'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
