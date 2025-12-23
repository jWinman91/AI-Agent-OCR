from typing import Literal

import streamlit as st
from src.backend_client.request_be import BeRequest


class AgentConfigurations:
    """
    Class to manage agent configurations in the Streamlit app.
    """

    RESET_BUTTON_STYLE = """
    <style>
    /* Style the most recently rendered Streamlit button (the reset button) */
    div.stButton > button:last-child {
        background-color: #ff4d4d;
        color: white;
        border: none;
    }
    div.stButton > button:last-child:hover {
        background-color: #e60000;
    }
    </style>
    """

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 8000,
        protocol: str = "http",
    ) -> None:
        """
        Initializes the AgentConfigurations with backend connection details.

        :param ip: IP address of the backend server.
        :param port: Port number of the backend server.
        :param protocol: Protocol to use for the request (default is "http").
        """
        self._request_be = BeRequest(ip, port, protocol)

    def build_configuration_form(
        self,
        agent_name: Literal["extractor", "plotter"],
    ) -> None:
        """
        Builds the agent configuration form for the Streamlit app.
        Allows users to configure agent settings (system prompt and model params).

        :param agent_name: Name of the agent ("extractor" or "plotter").
        """
        with st.form(f"agent_configuration_form_{agent_name}", clear_on_submit=True):
            st.subheader(f"{agent_name.capitalize()} Configuration")
            config = self._request_be.get(
                path="get_config",
                params={"config_name": agent_name},
                data_type=None,
            )

            system_prompt = st.text_area(
                "System Prompt",
                value=config.get("system_prompt_user", ""),
                height=300,
            )

            model_provider_preset = config.get("model_provider", "openai")
            model_provider = st.selectbox(
                "Model Provider",
                options=["openai", "ollama"],
                index=["openai", "ollama"].index(
                    model_provider_preset if model_provider_preset else "openai"
                ),
            )

            model = st.text_input(
                "Model",
                value=config.get(
                    "model", "gpt-5" if model_provider == "openai" else "qwen-7b"
                ),
            )

            if model_provider == "openai":
                api_key = st.text_input("API Key", type="password")

                # set the API key in the backend
                if api_key:
                    self._request_be.post(
                        "set_openai_api_key",
                        {"api_key": api_key},
                        payload_type="params",
                    )

            if "mcp_servers" in config and config["mcp_servers"] is not None:
                available_mcp_servers = self._request_be.get(path="get_mcp_servers")[
                    "mcp_servers"
                ]
                preset_names = [
                    p["args"][0].split("/")[-1] for p in config.get("mcp_servers", [])
                ]

                selected_mcp = st.multiselect(
                    "MCP Servers", options=available_mcp_servers, default=preset_names
                )
                mcp_servers = [
                    {"command": "python", "args": [f"src/server/{name}"]}
                    for name in selected_mcp
                ]
            else:
                mcp_servers = None

            submit_button = st.form_submit_button("Save Configuration")

            if submit_button:
                config_data = {
                    "system_prompt_user": system_prompt,
                    "model": model,
                    "model_provider": model_provider,
                    "mcp_server": mcp_servers,
                }

                success = self._request_be.post(
                    "upload_config",
                    agent_name,
                    config_data,
                )
                if success:
                    st.session_state[f"save_configuration_{agent_name}"] = True
                    st.rerun()
                else:
                    st.error("Failed to save configuration.")

            if st.session_state.get(f"save_configuration_{agent_name}", False):
                st.session_state[f"save_configuration_{agent_name}"] = False
                st.success(f"Configuration for {agent_name} saved successfully.")

    def build_reset_button(self, agent_name: Literal["extractor", "plotter"]) -> None:
        """
        Builds a reset button for the agent configuration.

        :param agent_name: Name of the agent ("extractor" or "plotter").
        """
        st.markdown(
            self.RESET_BUTTON_STYLE,
            unsafe_allow_html=True,
        )

        reset_button = st.button(f"Reset {agent_name.capitalize()} Configuration")
        if reset_button:
            success = self._request_be.post(
                "reset_config/",
                {"config_name": agent_name},
                payload_type="params",
            )
            if success:
                st.session_state[f"reset_configuration_{agent_name}"] = True
                st.rerun()
            else:
                st.error(f"Failed to reset {agent_name} configuration.")

        if st.session_state.get(f"reset_configuration_{agent_name}", False):
            st.session_state[f"reset_configuration_{agent_name}"] = False
            st.success(f"{agent_name.capitalize()} configuration reset successfully.")

    def build_page(self) -> None:
        """
        Builds the agent configuration page for the Streamlit app.
        """
        for agent_name in ["extractor", "plotter"]:
            with st.container(border=True):
                self.build_configuration_form(agent_name=agent_name)
                self.build_reset_button(agent_name=agent_name)


agent_configuration = AgentConfigurations()
agent_configuration.build_page()
