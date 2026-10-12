from tools.audio import TranscribeAudioTool
from tools.chess_tools import ChessBestMoveTool, ChessBoardFromImageTool
from tools.files import ReadFileTool
from tools.image import AnalyzeImageTool
from tools.python_runner import RunPythonFileTool
from tools.visit_webpage import VisitWebpageTool
from tools.web_search import WebSearchTool
from tools.wikipedia import WikipediaPageTool
from tools.youtube import YouTubeTranscriptTool, YouTubeVideoAnalysisTool


def get_tools():
    return [
        WebSearchTool(),
        VisitWebpageTool(),
        WikipediaPageTool(),
        YouTubeTranscriptTool(),
        YouTubeVideoAnalysisTool(),
        ReadFileTool(),
        RunPythonFileTool(),
        TranscribeAudioTool(),
        AnalyzeImageTool(),
        ChessBoardFromImageTool(),
        ChessBestMoveTool(),
    ]
