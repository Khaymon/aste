from aste.train.models.tasks import *


class ASTEModel:
    @staticmethod
    def get_model(name: str) -> BaseModel:
        if name == "BaseDiscriminatorModel":
            return BaseDiscriminatorModel
        elif name == "BaseGenerationModel":
            return BaseGenerativeModel
        else:
            raise NotImplementedError(f"Model {name} is not implemented")
