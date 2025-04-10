USE [master]
GO
/****** Object:  Database [CzechBankingAnalysis]    Script Date: 15-Mar-25 19:27:33 ******/
CREATE DATABASE [CzechBankingAnalysis]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'CzechBankingAnalysis', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.MSSQLSERVER\MSSQL\DATA\CzechBankingAnalysis.mdf' , SIZE = 139264KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'CzechBankingAnalysis_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.MSSQLSERVER\MSSQL\DATA\CzechBankingAnalysis_log.ldf' , SIZE = 139264KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT, LEDGER = OFF
GO
ALTER DATABASE [CzechBankingAnalysis] SET COMPATIBILITY_LEVEL = 160
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [CzechBankingAnalysis].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [CzechBankingAnalysis] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET ARITHABORT OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [CzechBankingAnalysis] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [CzechBankingAnalysis] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET  ENABLE_BROKER 
GO
ALTER DATABASE [CzechBankingAnalysis] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [CzechBankingAnalysis] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET RECOVERY FULL 
GO
ALTER DATABASE [CzechBankingAnalysis] SET  MULTI_USER 
GO
ALTER DATABASE [CzechBankingAnalysis] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [CzechBankingAnalysis] SET DB_CHAINING OFF 
GO
ALTER DATABASE [CzechBankingAnalysis] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [CzechBankingAnalysis] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [CzechBankingAnalysis] SET DELAYED_DURABILITY = DISABLED 
GO
ALTER DATABASE [CzechBankingAnalysis] SET ACCELERATED_DATABASE_RECOVERY = OFF  
GO
EXEC sys.sp_db_vardecimal_storage_format N'CzechBankingAnalysis', N'ON'
GO
ALTER DATABASE [CzechBankingAnalysis] SET QUERY_STORE = ON
GO
ALTER DATABASE [CzechBankingAnalysis] SET QUERY_STORE (OPERATION_MODE = READ_WRITE, CLEANUP_POLICY = (STALE_QUERY_THRESHOLD_DAYS = 30), DATA_FLUSH_INTERVAL_SECONDS = 900, INTERVAL_LENGTH_MINUTES = 60, MAX_STORAGE_SIZE_MB = 1000, QUERY_CAPTURE_MODE = AUTO, SIZE_BASED_CLEANUP_MODE = AUTO, MAX_PLANS_PER_QUERY = 200, WAIT_STATS_CAPTURE_MODE = ON)
GO
USE [CzechBankingAnalysis]
GO
/****** Object:  Table [dbo].[Account]    Script Date: 15-Mar-25 19:27:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Account](
	[account_id] [int] NOT NULL,
	[district_id] [int] NOT NULL,
	[frequency] [varchar](50) NOT NULL,
	[acc_date] [date] NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[account_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Client]    Script Date: 15-Mar-25 19:27:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Client](
	[client_id] [int] NOT NULL,
	[birth_number] [varchar](20) NOT NULL,
	[district_id] [int] NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[client_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CreditCard]    Script Date: 15-Mar-25 19:27:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CreditCard](
	[card_id] [smallint] NOT NULL,
	[disp_id] [smallint] NOT NULL,
	[type] [nvarchar](50) NOT NULL,
	[issued] [nvarchar](50) NOT NULL,
 CONSTRAINT [PK_CreditCard] PRIMARY KEY CLUSTERED 
(
	[card_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Demograph]    Script Date: 15-Mar-25 19:27:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Demograph](
	[A1] [int] NOT NULL,
	[A2] [nvarchar](100) NULL,
	[A3] [nvarchar](70) NULL,
	[A4] [float] NULL,
	[A5] [float] NULL,
	[A6] [int] NULL,
	[A7] [int] NULL,
	[A8] [int] NULL,
	[A9] [int] NULL,
	[A10] [float] NULL,
	[A11] [int] NULL,
	[A12] [float] NULL,
	[A13] [float] NULL,
	[A14] [int] NULL,
	[A15] [int] NULL,
	[A16] [int] NULL,
PRIMARY KEY CLUSTERED 
(
	[A1] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Disposition]    Script Date: 15-Mar-25 19:27:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Disposition](
	[disp_id] [smallint] NOT NULL,
	[client_id] [int] NULL,
	[account_id] [int] NULL,
	[type] [nvarchar](50) NOT NULL,
 CONSTRAINT [PK_Disposition] PRIMARY KEY CLUSTERED 
(
	[disp_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Loan]    Script Date: 15-Mar-25 19:27:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Loan](
	[Loan_id] [int] NOT NULL,
	[Account_id] [int] NULL,
	[Loan_date] [nvarchar](10) NULL,
	[Amount] [float] NULL,
	[Duration] [int] NULL,
	[Payments] [float] NULL,
	[Loan_status] [nvarchar](1) NULL,
PRIMARY KEY CLUSTERED 
(
	[Loan_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PermanentOrder]    Script Date: 15-Mar-25 19:27:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PermanentOrder](
	[Order_id] [int] NOT NULL,
	[Account_id] [int] NULL,
	[Bank_to] [nvarchar](10) NULL,
	[Account_to] [float] NULL,
	[Amount] [float] NULL,
	[K_symbol] [nvarchar](20) NULL,
PRIMARY KEY CLUSTERED 
(
	[Order_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Trans]    Script Date: 15-Mar-25 19:27:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Trans](
	[Trans_id] [int] NOT NULL,
	[Account_id] [int] NULL,
	[Trans_date] [nvarchar](10) NULL,
	[Trans_type] [nvarchar](10) NULL,
	[Operation] [nvarchar](100) NULL,
	[Amount] [float] NULL,
	[Balance] [float] NULL,
	[K_symbol] [nvarchar](20) NULL,
	[Bank] [nvarchar](20) NULL,
	[Account] [float] NULL,
PRIMARY KEY CLUSTERED 
(
	[Trans_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
ALTER TABLE [dbo].[Account]  WITH CHECK ADD  CONSTRAINT [FK_Account_Demograph] FOREIGN KEY([district_id])
REFERENCES [dbo].[Demograph] ([A1])
GO
ALTER TABLE [dbo].[Account] CHECK CONSTRAINT [FK_Account_Demograph]
GO
ALTER TABLE [dbo].[Client]  WITH CHECK ADD  CONSTRAINT [FK_Client_Demograph] FOREIGN KEY([district_id])
REFERENCES [dbo].[Demograph] ([A1])
GO
ALTER TABLE [dbo].[Client] CHECK CONSTRAINT [FK_Client_Demograph]
GO
ALTER TABLE [dbo].[CreditCard]  WITH CHECK ADD  CONSTRAINT [FK_CreditCard_Disposition] FOREIGN KEY([disp_id])
REFERENCES [dbo].[Disposition] ([disp_id])
GO
ALTER TABLE [dbo].[CreditCard] CHECK CONSTRAINT [FK_CreditCard_Disposition]
GO
ALTER TABLE [dbo].[Disposition]  WITH CHECK ADD  CONSTRAINT [FK_Disposition_Account] FOREIGN KEY([account_id])
REFERENCES [dbo].[Account] ([account_id])
GO
ALTER TABLE [dbo].[Disposition] CHECK CONSTRAINT [FK_Disposition_Account]
GO
ALTER TABLE [dbo].[Disposition]  WITH CHECK ADD  CONSTRAINT [FK_Disposition_Client] FOREIGN KEY([client_id])
REFERENCES [dbo].[Client] ([client_id])
GO
ALTER TABLE [dbo].[Disposition] CHECK CONSTRAINT [FK_Disposition_Client]
GO
ALTER TABLE [dbo].[Loan]  WITH CHECK ADD  CONSTRAINT [FK_Loan_Account] FOREIGN KEY([Account_id])
REFERENCES [dbo].[Account] ([account_id])
GO
ALTER TABLE [dbo].[Loan] CHECK CONSTRAINT [FK_Loan_Account]
GO
ALTER TABLE [dbo].[PermanentOrder]  WITH CHECK ADD  CONSTRAINT [FK_PermanentOrder_Account] FOREIGN KEY([Account_id])
REFERENCES [dbo].[Account] ([account_id])
GO
ALTER TABLE [dbo].[PermanentOrder] CHECK CONSTRAINT [FK_PermanentOrder_Account]
GO
ALTER TABLE [dbo].[Trans]  WITH CHECK ADD  CONSTRAINT [FK_Trans_Account] FOREIGN KEY([Account_id])
REFERENCES [dbo].[Account] ([account_id])
GO
ALTER TABLE [dbo].[Trans] CHECK CONSTRAINT [FK_Trans_Account]
GO
USE [master]
GO
ALTER DATABASE [CzechBankingAnalysis] SET  READ_WRITE 
GO
