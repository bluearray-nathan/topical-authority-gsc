# Project Instructions: Streamlit Cloud App

## Tech Stack
- Framework: Streamlit (latest)
- Environment: Streamlit Community Cloud
- Language: Python 3.11+

## Development Rules
1. **Secrets Management**: Never hardcode API keys or credentials. Always use `st.secrets["KEY_NAME"]`.
2. **Pathing**: Use `pathlib` for all file operations to ensure cross-platform compatibility (Mac local vs. Linux Cloud).
3. **Session State**: Use `st.session_state` for multi-page data persistence. 
4. **Performance**: Prioritize `@st.cache_data` for data loading and `@st.cache_resource` for database connections.
5. **UI Consistency**: Maintain the existing theme. Use `st.columns` for responsive layouts.

## Deployment Note
- This app is deployed via GitHub to Streamlit Cloud. 
- Ensure all new dependencies are automatically added to `requirements.txt`.