import nest_asyncio
import pins
from confection import Config
from dotenv import load_dotenv
from pyprojroot import here

config = Config().from_disk(here("project.cfg"))

# Load environment variables
load_dotenv()

# Make sure that asyncio works in Jupyter notebooks
# https://github.com/ipython/ipython/issues/11338
nest_asyncio.apply()

board = pins.board_s3(path=config["paths"]["s3_bucket"])
