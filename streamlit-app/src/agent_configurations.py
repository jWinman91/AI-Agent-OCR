from typing import Literal

import streamlit as st
from src.backend_client.request_be import BeRequest


class AgentConfigurations:
    """
    Class to manage agent configurations in the Streamlit app.
    """

    RESET_BUTTON_STYLE = """
    <style>
    /* Match ALL buttons whose label starts with "Reset" */
    button:has(span:contains("Reset")) {
        background-color: #ff4d4d;
        color: white;
        border: none;
        border-radius: 6px;
    }

    /* Hover state */
    button:has(span:contains("Reset")):hover {
        background-color: #e60000;
        color: white;
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
        st.markdown(
            self.RESET_BUTTON_STYLE,
            unsafe_allow_html=True,
        )

    def build_configuration_form(
        self,
        agent_name: Literal["data_extractor", "analyser"],
    ) -> None:
        """
        Builds the agent configuration form for the Streamlit app.
        Allows users to configure agent settings (system prompt and model params).

        :param agent_name: Name of the agent ("data_extractor" or "analyser").
        """
        if st.session_state.pop(f"pending_reset_{agent_name}", False):
            for prefix in (
                "system_prompt",
                "model_provider_selectbox",
            ):
                st.session_state.pop(f"{prefix}_{agent_name}", None)
            for widget_key in list(st.session_state):
                if widget_key.endswith(f"_model_selectbox_{agent_name}"):
                    st.session_state.pop(widget_key, None)
        st.subheader(f"{agent_name.capitalize()} Configuration")
        config = self._request_be.get(
            path="get_config",
            params={"config_name": agent_name},
            data_type=None,
        )
        print("Agent name: ", agent_name)
        print("Config: ", config)

        system_prompt = st.text_area(
            "System Prompt",
            value=config.get("system_prompt_user", ""),
            height=300,
            key=f"system_prompt_{agent_name}",
        )

        model_provider_preset = config.get("model_provider", "openai")
        model_providers = self._request_be.get(path="get_model_providers/")
        if not model_providers:
            st.error("No model providers available.")
            return
        model_provider = st.selectbox(
            "Model Provider",
            options=model_providers,
            index=model_providers.index(model_provider_preset)
            if model_provider_preset in model_providers
            else 0,
            key=f"model_provider_selectbox_{agent_name}",
        )

        model = None
        try:
            models = self._request_be.get(
                path="get_models/", params={"model_provider": model_provider}
            )
        except RuntimeError as error:
            st.error(str(error))
            models = []
        if models:
            saved_model = config.get("model")
            if saved_model not in models:
                st.warning("Configured model is unavailable. Select a replacement.")
            model_key = f"{model_provider}_model_selectbox_{agent_name}"
            if st.session_state.get(model_key) not in models:
                st.session_state.pop(model_key, None)
            model = st.selectbox(
                "Model",
                options=models,
                index=models.index(saved_model) if saved_model in models else 0,
                key=model_key,
            )
        else:
            st.warning(f"No models available for {model_provider}.")
        if model_provider == "openai":
            api_key = st.text_input(
                "API Key",
                type="password",
                key=f"api_key_{agent_name}",
            )

            # set the API key in the backend
            if api_key:
                self._request_be.post(
                    "set_openai_api_key",
                    {"api_key": api_key},
                    payload_type="params",
                )

        if st.button(
            "Save Configuration",
            type="secondary",
            key=f"save_configuration_button_{agent_name}",
            disabled=not model,
        ):
            config_data = {
                "system_prompt_user": system_prompt,
                "model": model,
                "model_provider": model_provider,
            }

            success = self._request_be.post(
                "upload_config",
                agent_name,
                config_data,
            )
            if success is True:
                st.session_state[f"save_configuration_{agent_name}"] = True
                st.rerun()
            else:
                st.error("Failed to save configuration.")

        if st.session_state.get(f"save_configuration_{agent_name}", False):
            st.session_state[f"save_configuration_{agent_name}"] = False
            st.success(f"Configuration for {agent_name} saved successfully.")

    def build_reset_button(
        self, agent_name: Literal["data_extractor", "analyser"]
    ) -> None:
        """
        Builds a reset button for the agent configuration.

        :param agent_name: Name of the agent ("data_extractor" or "analyser").
        """
        reset_button = st.button(
            "Reset Configuration",
            type="primary",
            key=f"reset_button_{agent_name}",
        )
        if reset_button:
            success = self._request_be.post(
                "reset_config/",
                {"config_name": agent_name},
                payload_type="params",
            )
            if success is True:
                st.session_state[f"pending_reset_{agent_name}"] = True
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
        for agent_name in ["data_extractor", "analyser"]:
            with st.container(border=True):
                try:
                    with st.container(border=True):
                        self.build_configuration_form(agent_name=agent_name)
                except Exception as error:
                    st.error(f"Cannot configure {agent_name}: {error}")
                try:
                    self.build_reset_button(agent_name=agent_name)
                except Exception as error:
                    st.error(f"Cannot reset {agent_name}: {error}")


agent_configuration = AgentConfigurations()
agent_configuration.build_page()
