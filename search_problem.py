from abc import ABC, abstractmethod
from typing import Any, List

class SearchProblem(ABC):

    @abstractmethod
    def initial_state(self) -> Any:
        pass

    @abstractmethod
    def goal_test(self, state: Any) -> bool:
        pass

    @abstractmethod
    def actions(self, state: Any) -> List[Any]:
        pass

    @abstractmethod
    def result(self, state: Any, action: Any) -> Any:
        pass

    @abstractmethod
    def step_cost(self, state: Any, action: Any, next_state: Any) -> float:
        pass
