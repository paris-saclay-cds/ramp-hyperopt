import csv
import json
import os
import subprocess
import time
import zipfile
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rampwf as rw
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import ramphy as rh
from ramphy import ramp_setup as rs


def execute_script(script_path: str | Path, env_args: Dict, script_args: Dict, hydra_args: Dict) -> bool:
    """executes the required python script

    Args:
        script_path (str | Path): _description_
        env_args (Dict): _description_
        script_args (Dict): _description_
        hydra_args (Dict): _description_

    Returns:
        bool: True if terminated properly, False otherwise
    """
    # Ensure script_path is a Path object
    script_path = str(script_path)

    # Add environment args
    env = os.environ.copy()
    for env_variable in env_args:
        env[env_variable] = str(env_args[env_variable])

    # Construct the command to run the script with its arguments
    cmd = ["python", script_path]
    for key, value in script_args.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    # Add any arguments for hydra (these are handled differently than script)
    for key, value in hydra_args.items():
        cmd.append(f"{key}={value} ")

    # Run the script
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    # Optionally, you can print or log the output
    print(result.stdout)
    print(result.stderr)
    return True


def find_best(submission: str, ramp_kit_dir: Path | str) -> str:
    submissions = os.listdir(str(Path(ramp_kit_dir) / "submissions"))
    for sub in submissions:
        if submission in sub:
            print(f"Best submission: {sub}")
            return sub
    raise ValueError(f"No hyperopted best submission {submission}")


def download_file(download_url: str, destination: Path, config_values: dict):
    """Downloads a file from the specified URL

    Saves the file to the provided destination path.

    Args:
        download_url (str): The URL to download the file from.
        destination (str): The local path to save the downloaded file.
        config_values (dict): Configuration for Kaggle, read from ~/.kaggle/kaggle.json
    """
    try:
        driver = 'firefox'

        if driver == 'chrome':
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--ignore-certificate-errors')
            chrome_options.add_argument('--ignore-ssl-errors')

            prefs = {
                "download.default_directory": destination.parent.absolute().as_posix(),
                "download.prompt_for_download": False,
                "directory_upgrade": True,
                "safebrowsing.enabled": False
            }
            chrome_options.add_experimental_option("prefs", prefs)
            chrome_driver_path = '/usr/bin/chromedriver'
            service = Service(executable_path=chrome_driver_path)
            driver = webdriver.Chrome(
                service=service, options=chrome_options,
                )
        else:  # firefox
            from selenium.webdriver import FirefoxOptions
            options = FirefoxOptions()
            options.add_argument("--headless")

            driver = webdriver.Firefox(
                options=options)

        login_url = 'https://www.kaggle.com/account/login?phase=emailSignIn'
        driver.get(login_url)
        if driver.page_source == '<html><head></head><body></body></html>':
            raise ValueError("Issue loading the page, Most likely a Proxy error.")
        username_field = driver.find_element(By.ID, ':r0:')
        username_field.send_keys(config_values['login_email'])
        password_field = driver.find_element(By.ID, ':r1:')
        password_field.send_keys(config_values['login_password'])
        login_button = driver.find_element(By.XPATH, "//button[span[text()='Sign In']]")
        driver.execute_script("arguments[0].scrollIntoView(true);", login_button)
        login_button.click()
        time.sleep(1)

        # relying on requests as it could be tricky, even if feasible, to make sure the
        # download ended before terminating the script.
        # passing cookies to request
        cookies = driver.get_cookies()
    except Exception as e:
        print(f"{e}")
    finally:
        driver.quit()

    session_cookies = {cookie['name']: cookie['value'] for cookie in cookies}
    response = requests.get(
        download_url, stream=True, verify=False, cookies=session_cookies)
    response.raise_for_status()  # raises an HTTPError for bad responses

    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print("Download complete. File saved to:", destination)

    # download with selenium
    # driver.get(download_url)
    # download_button = WebDriverWait(driver, 10).until(
    #     EC.presence_of_element_located((By.XPATH, '//button[@title="Download Leaderboard"]'))
    # )
    # driver.execute_script("arguments[0].click();", download_button)
    # time.sleep(1)


def download_leaderboard(
    kaggle_api: KaggleApi,
    competition: str,
    zip_destination: Path,
    phase: str
):
    """Downloads leaderboard.

    This is not supported by the Kaggle API so we implement it ourselves as there is
    a link on the leaderboard webpage to download it.

    Args:
        kaggle_api (object): KaggleAPI instance
        competition (str): Identifier of the competition
        zip_destination (Path): where the zip file is
        phase (str): public or private
    """
    # the url that is required to download the leaderboard needs the numeric
    # identifier of the competition, we first retrieve it.
    # using 'default' for the group parameter only returns the active competitions so
    # we use the entered one.
    result = kaggle_api.process_response(
        kaggle_api.competitions_list_with_http_info(
            group="entered", category="", sort_by="", page=1, search=competition
        )
    )
    # there can be several competitions satisfying the search query, we pick the one
    # we are looking for using the unique competition url
    competition_url = f"https://www.kaggle.com/competitions/{competition}"
    for competition in result:
        if competition["url"] == competition_url:
            break
    num_id = competition["id"]
    download_file(
        f"https://www.kaggle.com/competitions/{num_id}/leaderboard/download/{phase}",
        zip_destination, kaggle_api.config_values
    )


def read_leaderboard_scores(
    competition: str,
    zip_file: Path,
    destination_folder: Path,
    phase: str,
) -> np.ndarray:
    """Read leaderboard scores from file.

    Args:
        competition (str): Identifier of the competition
        zip_file (Path): Zip file containing the leaderboard
        destination_folder (Path): Where to extract the zip file contents.
        phase (str): public or private

    Returns:
        scores (np.array): Leaderboard scores.
    """
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(destination_folder)

    # the file contains the download date in its name so we consider the case where
    # there are several of them. they should all be the same.
    files = list(destination_folder.glob(f"{competition}-{phase}leaderboard-*.csv"))
    leaderboard_file = files[0]
    return pd.read_csv(leaderboard_file)["Score"].to_numpy()


def get_leaderboard_scores(
    kaggle_api: KaggleApi,
    competition: str,
    phase: str,
) -> np.ndarray:
    """Get pleaderboard scores from Kaggle.

    Args:
        kaggle_api (object): KaggleAPI instance.
        competition (str): Kaggle competition name.
        phase (str): public or private

    Returns:
        scores (np.array): leaderboard scores.
            Sorted from best to worst.
    """
    destination_folder = Path(f"{phase}_leaderboard_{competition}")
    zip_destination = destination_folder / f"{phase}_leaderboard_{competition}.zip"
    if not zip_destination.exists():
        destination_folder.mkdir(exist_ok=True)
        download_leaderboard(kaggle_api, competition, zip_destination, phase)
    scores = read_leaderboard_scores(
        competition,
        zip_destination,
        destination_folder,
        phase,
    )
    return scores


def get_submission_scores(kaggle_api: KaggleApi, competition: str) -> Tuple[float, float]:
    """Get submission scores from Kaggle.

    This takes the scores of the most recent submission.

    Args:
        kaggle_api (object): KaggleApi instance.
        competition (str): Kaggle competition name.

    Returns:
        public_score, private_score (float): Public and private scores of the submission
    """
    # XXX this should be improved to make sure we retrieve the submission we are
    # interested in (using an unique identifier or something like this).
    # here we assume that our submission is the most recent one.
    status = "pending"  # the last submission is still being evaluated by Kaggle
    max_retry = 30
    n_retries = 0
    while status == "pending":
        # submissions are listed from most recent to oldest.
        submission_kaggle_id = kaggle_api.competition_submissions(competition=competition)[0]
        status = kaggle_api.string(getattr(submission_kaggle_id, "status"))
        if status == "pending":
            if n_retries < max_retry:
                n_retries += 1
                time.sleep(10)
            else:
                raise RuntimeError(
                    f"Maximum number of retries ({max_retry}) exceeded and submission "
                    "is still being evaluated by Kaggle. Consider increasing max_retry"
                )

    public_score = kaggle_api.string(getattr(submission_kaggle_id, "publicScore"))
    public_score = float(public_score)
    private_score = kaggle_api.string(getattr(submission_kaggle_id, "privateScore"))
    private_score = float(private_score)

    # use the following if you need to have access to all the submissions and process
    # all the results
    # it is adapted from kaggle_api.print_csv
    # fields = [
    #     'fileName', 'date', 'description', 'status', 'publicScore', 'privateScore'
    # ]
    # csv_buffer = StringIO()
    # writer = csv.writer(csv_buffer)
    # writer.writerow(fields)
    # for i in submissions_raw:
    #     i_fields = [kaggle_api.string(getattr(i, f)) for f in fields]
    #     writer.writerow(i_fields)
    # csv_buffer.seek(0)
    # df = pd.read_csv(csv_buffer)

    return public_score, private_score


@rh.actions.ramp_action
def kaggle_submit(
    submission: str,
    ramp_kit_dir: Path | str,
    competition: str,
    message: Optional[str] = None,
) -> Dict:
    """Submits the predictions to Kaggle

    Args:
        submission (str): RAMP Submission name. If blended it submits the blended results
        ramp_kit_dir (Path | str): Path to ramp kit
        competition (str): Kaggle competition identifier name
        message (Optional[str], optional): Description to Kaggle submission. Defaults to None.

    Returns:
        Dict: _description_
    """
    problem = rw.utils.assert_read_problem(ramp_kit_dir)
    is_lower_the_better = problem.score_types[0].is_lower_the_better

    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    action_output = {}

    private_scores = get_leaderboard_scores(kaggle_api, competition, 'private')
    public_scores = get_leaderboard_scores(kaggle_api, competition, 'public')

    if submission == "blended":
        file_path = Path(ramp_kit_dir) / "submissions" / "training_output" / "submission_combined_bagged_test.csv"
        assert file_path.exists(), "No blended test data found."
        message = f"Blended {Path(ramp_kit_dir).name}"
        submission_status = kaggle_api.competition_submit(file_name=file_path, message=message, competition=competition)
    elif "_best_0_" in submission:
        submission = find_best(submission=submission, ramp_kit_dir=ramp_kit_dir)

        if message is None:
            message = submission

        file_path = Path(ramp_kit_dir) / "submissions" / submission / "training_output" / "submission_bagged_test.csv"
        assert Path(ramp_kit_dir) / "submissions" / submission, f"Submission {submission} does not exists."
        assert file_path.exists(), f"File {file_path} does not exists. Sure that the submission has been trained?"
        submission_status = kaggle_api.competition_submit(file_name=file_path, message=message, competition=competition)
    else:  # submission is a path
        if message is None:
            message = ""

        file_path = Path(submission)

        submission_status = kaggle_api.competition_submit(file_name=file_path, message=message, competition=competition)

    action_output["kaggle_submission_status"] = submission_status
    action_output["kaggle_submission_message"] = message

    # get private and public scores of the submission
    public_score, private_score = get_submission_scores(kaggle_api, competition)

    if is_lower_the_better:
        public_rank = np.mean(public_scores > public_score)
        private_rank = np.mean(private_scores > private_score)
    else:
        public_rank = np.mean(public_scores < public_score)
        private_rank = np.mean(private_scores < private_score)

    action_output["public_score"] = public_score
    action_output["private_score"] = private_score
    action_output["public_rank"] = public_rank
    action_output["private_rank"] = private_rank

    return action_output
