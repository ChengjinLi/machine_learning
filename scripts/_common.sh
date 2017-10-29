source ${PWD}/_project_env.sh

WORKSPACE=$(cd ${PWD}/..; printf "$PWD")

ENV_PATH="${WORKSPACE}/env_${PROJECT_NAME}"
